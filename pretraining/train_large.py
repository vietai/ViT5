import tensorflow
import functools
import os
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow.compat.v1 as tf
import gin
from t5 import models
import t5
import gin
import subprocess
from random import shuffle

print(tensorflow.__version__)

ON_CLOUD = True

if ON_CLOUD:
  print("Setting up GCS access...")
  import tensorflow_gcs_config
  from google.colab import auth
  # Set credentials for GCS reading/writing from Colab and TPU.
  TPU_TOPOLOGY = "v2-8"
  try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU zdetection
    TPU_ADDRESS = tpu.get_master()
    print('Running on TPU:', TPU_ADDRESS)
  except ValueError:
    raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')
  auth.authenticate_user()
  tf.config.experimental_connect_to_host(TPU_ADDRESS)
  tensorflow_gcs_config.configure_gcs_from_colab_auth()

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

gin.parse_config_file(
        '../configs/large_operative_config.gin'
    )

def dumping_dataset(split, shuffle_files = False):
    del shuffle_files
    files_name_cc100 = list(map(lambda x: x.strip(), subprocess.run(['gsutil', 'ls', 'gs://vie_projects/data/cc100_1024/cc100*.txt'], stdout=subprocess.PIPE).stdout.splitlines()))

    shuffle(files_name_cc100)

    print(files_name_cc100[0])

    ds = tf.data.TextLineDataset(
       files_name_cc100
    )
    ds = ds.map(lambda *ex: dict(zip(['title', 'text'], ['None',ex[0]])))
    ds = ds.shuffle(buffer_size=1000000)

    return ds

vocab = "gs://t5_training/models/spm/vie/wiki_vietnamese_vocab.model"
t5.data.TaskRegistry.remove('dumping_dataset')
t5.data.TaskRegistry.add(
    'dumping_dataset',
    dataset_fn = dumping_dataset,
    splits = ['train'],
    text_preprocessor =  functools.partial(
        t5.data.preprocessors.rekey,
        key_map = {'inputs': None, 'targets': 'text'},
    ),
    token_preprocessor = t5.data.preprocessors.unsupervised,
    output_features=t5.data.Feature(vocabulary=t5.data.SentencePieceVocabulary(vocab)),
    metric_fns = [],
)



t5.data.MixtureRegistry.remove('all_vieT5')
t5.data.MixtureRegistry.add(
    'all_vieT5',
    [
        'dumping_dataset',
    ],
    default_rate = 1.0,
)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


MODEL_SIZE = 'large'
model_parallelism, train_batch_size, keep_checkpoint_max = {
    'small': (1, 256, 16),
    'base': (2, 128, 8),
    'large': (8, 128, 4),
    '3B': (8, 16, 1),
    '11B': (8, 16, 1),
}[MODEL_SIZE]

model_dir = f'gs://t5_training/models/vie/viT5_1024/{MODEL_SIZE}'

model = models.MtfModel(
  model_dir = model_dir,
  tpu = TPU_ADDRESS,
  tpu_topology = TPU_TOPOLOGY,
  model_parallelism = model_parallelism,
  batch_size = train_batch_size,
  sequence_length = {'inputs': 1024, 'targets': 1024},
  learning_rate_schedule = 0.001,
  save_checkpoints_steps = 2000,
  keep_checkpoint_max = 5,
  iterations_per_loop = 100,
)

model.train(mixture_or_task_name = 'all_vieT5', steps = 1000000)