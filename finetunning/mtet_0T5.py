import tensorflow
import functools
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import argparse
import tensorflow.compat.v1 as tf
import gin
import t5
import os
from t5.models import MtfModel
from datasets import load_metric

print(tensorflow.__version__)


parser = argparse.ArgumentParser(description='Finetunning ViT5')
parser.add_argument('-tpu', dest='tpu', type=str, help='tpu address', default='0.0.0.0')
parser.add_argument('-task', dest='task', type=str, help='En to Vi(envi) or Vi to En(vien) task', default='envi')
parser.add_argument('-eval', dest='eval', type=str, help='Eval test set', default='tst')

parser.add_argument('-steps', dest='steps', type=int, help='tpu address', default=16266)
args = parser.parse_args()

TPU_TOPOLOGY = 'v2-8'
TPU_ADDRESS = args.tpu
TPU_ADDRESS = f'grpc://{TPU_ADDRESS}:8470'

print(f"TPU Address {TPU_ADDRESS}")
print(f"FINE TUNE STEPS {args.steps}")
ON_CLOUD = True

if ON_CLOUD:
  print("Setting up GCS access...")
  # import tensorflow_gcs_config
  from google.colab import auth
  # Set credentials for GCS reading/writing from Colab and TPU.
  TPU_TOPOLOGY = "v3-8"
  # auth.authenticate_user()
  tf.config.experimental_connect_to_host(TPU_ADDRESS)
  # tensorflow_gcs_config.configure_gcs_from_colab_auth()

tf.disable_v2_behavior()
gin.parse_config_file(
        '../configs/t5/base_operative_config.gin'
    )
# Improve logging.
from contextlib import contextmanager
import logging as py_logging

if ON_CLOUD:
  tf.get_logger().propagate = False
  py_logging.root.setLevel('INFO')

@contextmanager
def tf_verbosity_level(level):
  og_level = tf.logging.get_verbosity()
  tf.logging.set_verbosity(level)
  yield
  tf.logging.set_verbosity(og_level)



tf.disable_v2_behavior()

# Improve logging.
from contextlib import contextmanager
import logging as py_logging


@contextmanager
def tf_verbosity_level(level):
  og_level = tf.logging.get_verbosity()
  tf.logging.set_verbosity(level)
  yield
  tf.logging.set_verbosity(og_level)

task = args.task
vocab = f"gs://vien-translation/checkpoints/spm/cc100_envi_vocab.model"
def dumping_dataset(split, shuffle_files = False):
    del shuffle_files
    ds = tf.data.TextLineDataset(
        [
        f'gs://vien-translation/data/mtet/train_{task}_filtered.tsv',
        ]
        )
    ds = ds.map(
        functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                          field_delim="\t", use_quote_delim=False),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(lambda *ex: dict(zip(["input", "target"], ex)))
    ds = ds.shuffle(buffer_size=1000000)
    return ds

def translate_preprocessor(ds):
  def normalize_text(text):
    """Lowercase and remove quotes from a TensorFlow string."""
    return text

  def to_inputs_and_targets(ex):
    """Map {"inputs": ..., "targets": ...}->{"inputs": ner..., "targets": ...}."""
    return {
        "inputs":
            tf.strings.join(
                [f"{task}: ", normalize_text(ex["input"])]),
        "targets": normalize_text(ex["target"])
    }
  return ds.map(to_inputs_and_targets, 
                num_parallel_calls=tf.data.experimental.AUTOTUNE)


t5.data.TaskRegistry.remove(task)
t5.data.TaskRegistry.add(
    task,
    dataset_fn=dumping_dataset,
    splits=['train'],
    text_preprocessor=[translate_preprocessor],
    metric_fns=[],
    output_features=t5.data.Feature(vocabulary=t5.data.SentencePieceVocabulary(vocab))
)

t5.data.MixtureRegistry.remove("mtet_all")
t5.data.MixtureRegistry.add(
    "mtet_all",
    [task],
     default_rate=1.0,
)


MODEL_SIZE = "base"

# Set parallelism and batch size to fit on v3-8 TPU (if possible).
model_parallelism, train_batch_size, keep_checkpoint_max = {
    "small": (1, 256, 16),
    "base": (4, 256, 8),
    "large": (8, 256, 4),
    "3B": (8, 16, 1),
    "11B": (8, 16, 1)}[MODEL_SIZE]


# PRETRAINED_DIR = f"gs://vien-translation/checkpoints/enviT5_finetune/0T5/"

MODEL_DIR = f"gs://vien-translation/checkpoints/enviT5_finetune/mtet_{task}_0T5"

tf.io.gfile.makedirs(MODEL_DIR)
# The models from paper are based on the Mesh Tensorflow Transformer.

model = MtfModel(
    model_dir=MODEL_DIR,
    tpu=TPU_ADDRESS,
    tpu_topology=TPU_TOPOLOGY,
    model_parallelism=model_parallelism,
    batch_size=train_batch_size,
    sequence_length={"inputs": 128, "targets": 128},
    learning_rate_schedule=0.005,
    save_checkpoints_steps=2000,
    keep_checkpoint_max=keep_checkpoint_max if ON_CLOUD else None,
    # iterations_per_loop=100,
)

FINETUNE_STEPS = args.steps

# model.finetune(
#     mixture_or_task_name="mtet_all",
#     pretrained_model_dir=PRETRAINED_DIR,
#     finetune_steps=FINETUNE_STEPS
# )

model.train(mixture_or_task_name = 'mtet_all', steps = FINETUNE_STEPS)

eval = args.eval
if eval == 'tst':
  input_file = f'tst2013.{task[0:2]}.unfix'
  output_file = f'{task}_predict_output.txt'
  label_file = f"tst2013.{task[2:4]}.unfix"
elif eval =='phomt':
  input_file = f'test.{task[0:2]}'
  output_file = f'{task}_predict_output.txt'
  label_file = f'test.{task[2:4]}'
      
with open('predict_input.txt', 'w') as out:
  for line in open(f'../data/{eval}/{input_file}'):
    out.write(f"{task[0:2]}: {line}")

predict_inputs_path = 'predict_input.txt'
predict_outputs_path = output_file
# Manually apply preprocessing by prepending "triviaqa question:".


checkpoints = [int(x.replace('.index', '').split('-')[-1]) for x in tf.io.gfile.glob(MODEL_DIR +'/*ckpt*.index')]
results = []
for checkpoint in checkpoints:
  print('starting checkpoint', checkpoint)
  # Ignore any logging so that we only see the model's answers to the questions.
  with tf_verbosity_level('ERROR'):
    model.batch_size = 8  # Min size for small model on v2-8 with parallelism 1.
    model.predict(
        input_file=predict_inputs_path,
        output_file=predict_outputs_path,
        # Select the most probable output token at each step.
        vocabulary=t5.data.SentencePieceVocabulary(vocab),
        checkpoint_steps=checkpoint,
        temperature=0,
    )



  # The output filename will have the checkpoint appended so we glob to get 
  # the latest.
  prediction_files = sorted(tf.io.gfile.glob(predict_outputs_path + "*"))

  predictions = []
  references = []
  with open(f'../data/{eval}/{label_file}') as file:
    for line in file:
      references.append([f"{task[2:4]}: {line.strip()}"])
  with open(f'{output_file}-{checkpoint}') as file:
    for line in file:
      predictions.append(line.strip())

  # print('DEBUG: few senctences of pred')
  # print(predictions[0:3])

  # print('DEBUG: few senctences of ref')
  # print(references[0:3])
  metric = load_metric("sacrebleu", keep_in_memory=True)
  result = metric.compute(predictions=predictions, references=references)
  result = {"bleu": result["score"]}
  results.append([int(checkpoint), result])


sorted(results, key=lambda l:l[0])
print(results)
with open('result.tsv', 'w') as file:
  for result in results:
    file.write(f'{result[0]}\t{result[1]}\n')
