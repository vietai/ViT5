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
from data import files_name
from t5.models import MtfModel

print(tensorflow.__version__)


parser = argparse.ArgumentParser(description='Finetunning ViT5')
parser.add_argument('-tpu', dest='tpu', type=str, help='tpu address', default='0.0.0.0')
parser.add_argument('-steps', dest='steps', type=int, help='tpu address', default=0)
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
vocab = "gs://t5_training/models/spm/vie/wiki_vietnamese_vocab.model"

vocab = f"gs://translationv2/models/viT5_1024_large/wiki_vietnamese_vocab.model"
def dumping_dataset(split, shuffle_files = False):
    del shuffle_files
    if split == 'train':
      ds = tf.data.TextLineDataset(
            [
            'gs://t5_training/t5-data/vi_data/PhoNER/pho_ner_train.tsv',
            # 'gs://t5_training/t5-data/vi_data/PhoNER/pho_ner_dev.tsv',
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
  print(ex['target'].decode("utf-8"))
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


MODEL_SIZE = "large"

# Set parallelism and batch size to fit on v3-8 TPU (if possible).
model_parallelism, train_batch_size, keep_checkpoint_max = {
    "small": (1, 256, 16),
    "base": (8, 128, 8),
    "large": (8, 256, 4),
    "3B": (8, 16, 1),
    "11B": (8, 16, 1)}[MODEL_SIZE]
PRETRAINED_DIR = "gs://t5_training/models/vie/viT5_large_1024/"

MODEL_DIR = f"gs://t5_training/models/vie/PhoNER_viT5_large_1024"

tf.io.gfile.makedirs(MODEL_DIR)
# The models from paper are based on the Mesh Tensorflow Transformer.

model = MtfModel(
    model_dir=MODEL_DIR,
    tpu=TPU_ADDRESS,
    tpu_topology=TPU_TOPOLOGY,
    model_parallelism=model_parallelism,
    batch_size=train_batch_size,
    sequence_length={"inputs": 512, "targets": 512},
    learning_rate_schedule=0.001,
    save_checkpoints_steps=500,
    keep_checkpoint_max=keep_checkpoint_max if ON_CLOUD else None,
    iterations_per_loop=100,
)

FINETUNE_STEPS = args.steps

model.finetune(
    mixture_or_task_name="ner_all",
    pretrained_model_dir=PRETRAINED_DIR,
    finetune_steps=FINETUNE_STEPS
)