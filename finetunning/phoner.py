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

print(tensorflow.__version__)


parser = argparse.ArgumentParser(description='Finetunning ViT5')
parser.add_argument('-tpu', dest='tpu', type=str, help='tpu address', default='0.0.0.0')
parser.add_argument('-steps', dest='steps', type=int, help='finetune steps', default=45000)
parser.add_argument('-model_size', dest='model_size', type=str, help='Model Size', default='base')

args = parser.parse_args()

TPU_TOPOLOGY = 'v3-8'
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

task = 'phoner'
vocab = f"gs://translationv2/models/viT5_1024_large/wiki_vietnamese_vocab.model"
def dumping_dataset(split, shuffle_files = False):
    del shuffle_files
    if split == 'train':
      ds = tf.data.TextLineDataset(
            [
            'gs://translationv2/data/PhoNER/pho_ner_train.tsv',
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


t5.data.TaskRegistry.remove(task)
t5.data.TaskRegistry.add(
    task,
    dataset_fn=dumping_dataset,
    splits=["train", "validation"],
    text_preprocessor=[ner_preprocessor],
    metric_fns=[],
    output_features=t5.data.Feature(vocabulary=t5.data.SentencePieceVocabulary(vocab))
)

t5.data.MixtureRegistry.remove("ner_all")
t5.data.MixtureRegistry.add(
    "ner_all",
    [task],
     default_rate=1.0,
)


MODEL_SIZE = args.model_size

# Set parallelism and batch size to fit on v3-8 TPU (if possible).
model_parallelism, train_batch_size, keep_checkpoint_max = {
    "small": (1, 256, 16),
    "base": (4, 256, 8),
    "large": (8, 256, 4),
    "3B": (8, 16, 1),
    "11B": (8, 16, 1)}[MODEL_SIZE]


# set to True to use public checkpoint, else for internal vietAI
vietai_public_checkpoint = False
if vietai_public_checkpoint:
  PRETRAINED_DIR = f"gs://vietai_public/viT5/viT5_1024_{MODEL_SIZE}/"
else:
  PRETRAINED_DIR = f"gs://translationv2/models/viT5_1024_{MODEL_SIZE}/"

MODEL_DIR = f"gs://translationv2/models/viT5_finetune/PhoNER_viT5_large_1024"

tf.io.gfile.makedirs(MODEL_DIR)
# The models from paper are based on the Mesh Tensorflow Transformer.

model = MtfModel(
    model_dir=MODEL_DIR,
    tpu=TPU_ADDRESS,
    tpu_topology=TPU_TOPOLOGY,
    model_parallelism=model_parallelism,
    batch_size=train_batch_size,
    sequence_length={"inputs": 512, "targets": 512},
    learning_rate_schedule=0.0005,
    iterations_per_loop=100,
    save_checkpoints_steps=2000,
    keep_checkpoint_max=200,
)

FINETUNE_STEPS = args.steps

model.finetune(
    mixture_or_task_name="ner_all",
    pretrained_model_dir=PRETRAINED_DIR,
    finetune_steps=FINETUNE_STEPS
)


tasks = [
         ['PhoNER', "pho_ner"], 
    ]


import os
checkpoints = [int(x.replace('.index', '').split('-')[-1]) for x in tf.io.gfile.glob(MODEL_DIR +'/*ckpt*.index')]
print(checkpoints)


for checkpoint in checkpoints:
  for t in tasks:
    dir = t[0]
    task = t[1]

    
    input_file = task + '_test_input.txt'
    output_file = 'predict_output.txt'

    # Write out the supplied questions to text files.
    os.system(f"gsutil cp {os.path.join('gs://translationv2/data', dir, input_file)} .") 

    with open('predict_input.txt', 'w') as out:
      for line in open(input_file):
        line = line.replace('pho_ner', 'phoner')
        out.write(line)
        

    predict_inputs_path = 'predict_input.txt'
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
          checkpoint_steps=checkpoint,
          temperature=0,
      )

    # The output filename will have the checkpoint appended so we glob to get 
    # the latest.
    prediction_files = sorted(tf.io.gfile.glob(predict_outputs_path + "*"))
    print("Predicted task : " + task)
    print("\nPredictions using checkpoint %s:\n" % checkpoint)
    # with tf.io.gfile.GFile(prediction_files[-1]) as f:
    #   for q, a in zip(questions, f):
    #     if q:
    #       print("Q: " + q)
    #       print("A: " + a)
    #       print()
