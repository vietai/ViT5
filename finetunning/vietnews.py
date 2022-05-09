from tabnanny import check
import tensorflow
import functools
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import argparse
import tensorflow.compat.v1 as tf
import gin
from t5 import models
import t5
import gin
from random import shuffle
from t5.models import MtfModel
import tensorflow_datasets as tfds
from datasets import load_metric

print(tensorflow.__version__)


parser = argparse.ArgumentParser(description='Finetunning ViT5')
parser.add_argument('-tpu', dest='tpu', type=str, help='tpu address', default='0.0.0.0')
parser.add_argument('-steps', dest='steps', type=int, help='finetune steps', default=100000)
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
  # Set credentials for GCS reading/writing from Colab and TPU.
  TPU_TOPOLOGY = "v2-8"
  # auth.authenticate_user()
  tf.config.experimental_connect_to_host(TPU_ADDRESS)
  # tensorflow_gcs_config.configure_gcs_from_colab_auth()

tf.disable_v2_behavior()

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

task = 'vietnews'
vocab = f"gs://vien-translation/checkpoints/viT5_large_1024/wiki_vietnamese_vocab.model"
def dumping_dataset(split, shuffle_files = False):
    del shuffle_files
    if split == 'train':
      ds = tf.data.TextLineDataset(
            [
            'gs://vien-translation/data/vietnews/train_dedup.tsv',
            ]
          )
    else:
      ds = tf.data.TextLineDataset(
            [
                        ]
          )
    ds = ds.map(
        functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                          field_delim="\t", use_quote_delim=False),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(lambda *ex: dict(zip(["input", "target"], ex)))
    return ds

def ner_preprocessor(ds):
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
print("A few raw validation examples...")
for ex in tfds.as_numpy(dumping_dataset("train").take(5)):
  # print(base64.b64encode(ex['text']))
  print(ex['input'].decode("utf-8"), ex['target'].decode("utf-8"))
t5.data.TaskRegistry.remove(task)
t5.data.TaskRegistry.add(
    task,
    dataset_fn=dumping_dataset,
    splits=["train", "validation"],
    text_preprocessor=[ner_preprocessor],
    metric_fns=[],
    output_features=t5.data.Feature(vocabulary=t5.data.SentencePieceVocabulary(vocab))
)

t5.data.MixtureRegistry.remove("all")
t5.data.MixtureRegistry.add(
    "all",
    [task],
     default_rate=1.0,
)

# MODEL_NAME = "wiki_and_news_base"
length = 1024
MODEL_SIZE = "large"
# PRETRAINED_DIR = f'gs://t5_training/models/vie/viT5_1024/{MODEL_SIZE}'
# PRETRAINED_DIR = "gs://t5_training/models/vie/viT5_large_1024/"
PRETRAINED_DIR = "gs://vien-translation/checkpoints/viT5_large_1024"

MODEL_DIR = f"gs://vien-translation/checkpoints/viT5_finetune/vietnews_viT5_large"

tf.io.gfile.makedirs(MODEL_DIR)
# The models from paper are based on the Mesh Tensorflow Transformer.

# Set parallelism and batch size to fit on v2-8 TPU (if possible).
# Limit number of checkpoints to fit within 5GB (if possible).
model_parallelism, train_batch_size, keep_checkpoint_max = {
    "small": (1, 256, 16),
    "base": (4, 128, 8),
    "large": (8, 128, 4),
    "3B": (8, 16, 1),
    "11B": (8, 16, 1)}[MODEL_SIZE]

# The models from paper are based on the Mesh Tensorflow Transformer.
model = MtfModel(
    model_dir=MODEL_DIR,
    tpu=TPU_ADDRESS,
    tpu_topology=TPU_TOPOLOGY,
    model_parallelism=model_parallelism,
    batch_size=train_batch_size,
    sequence_length={"inputs": 1024, "targets": 256},
    learning_rate_schedule=0.001,
    save_checkpoints_steps=50000,
    keep_checkpoint_max=keep_checkpoint_max if ON_CLOUD else None,
    iterations_per_loop=100,
)
FINETUNE_STEPS = args.steps

model.finetune(
    mixture_or_task_name="all",
    pretrained_model_dir=PRETRAINED_DIR,
    finetune_steps=FINETUNE_STEPS
)


tasks = [
         ['vietnews', "vietnews"], 
        ]

for t in tasks:
  dir = t[0]
  task = t[1]

  
  input_file = 'predict_input.txt'
  output_file = 'predict_output.txt'

  # Write out the supplied questions to text files.
      

  predict_inputs_path = f'../data/{task}/{input_file}'
  predict_outputs_path = output_file
  # Manually apply preprocessing by prepending "triviaqa question:".

  # Ignore any logging so that we only see the model's answers to the questions.
  with tf_verbosity_level('ERROR'):
    model.batch_size = 8  # Min size for small model on v2-8 with parallelism 1.
    model.predict(
        input_file=predict_inputs_path,
        output_file=predict_outputs_path,
        # Select the most probable output token at each step.
        vocabulary=t5.data.SentencePieceVocabulary(vocab),
        temperature=0,
        checkpoint_steps=-1
    )

checkpoints = [int(x.replace('.index', '').split('-')[-1]) for x in tf.io.gfile.glob(MODEL_DIR +'/*ckpt*.index')]

metric = load_metric("rouge")

r2 = []
r1 = []
rL = []

checkpoint = checkpoints[-1]
score = metric.compute(predictions=open(f'predict_output.txt-{checkpoint}').readlines(), references=open('../data/vietnews/actual_output.txt').readlines())
# score = metric.compute()

rL.append([score['rougeL'].mid.fmeasure, score['rougeL'].mid.recall, score['rougeL'].mid.precision, checkpoint])
r2.append([ score['rouge2'].mid.fmeasure, score['rouge2'].mid.recall, score['rouge2'].mid.precision, checkpoint])
r1.append([score['rouge1'].mid.fmeasure, score['rouge1'].mid.recall, score['rouge1'].mid.precision, checkpoint])

r1.sort(key=lambda x: -x[0])
print(f'Checkpoint : {r1[0][3]}')
print(f"F1 : {r1[0][0]}")
print(f"recall : {r1[0][1]}")
print(f"precision: {r1[0][2]}")


r2.sort(key=lambda x: -x[0])
print(f'Checkpoint : {r2[0][3]}')
print(f"F1 : {r2[0][0]}")
print(f"recall : {r2[0][1]}")
print(f"precision: {r2[0][2]}")



rL.sort(key=lambda x: -x[0])
print(f'Checkpoint : {rL[0][3]}')
print(f"F1 : {rL[0][0]}")
print(f"recall : {rL[0][1]}")
print(f"precision: {rL[0][2]}")