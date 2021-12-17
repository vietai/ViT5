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
from data import files_name

print(tensorflow.__version__)

TPU_TOPOLOGY = 'v3-8'
TPU_ADDRESS = '	10.79.121.146'
TPU_ADDRESS = f'grpc://{TPU_ADDRESS}:8470'

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

gin.parse_config_file(
        '../configs/large_operative_config.gin'
    )

def dumping_dataset(split, shuffle_files = False):
    del shuffle_files
    files_name_cc100 = files_name

    shuffle(files_name_cc100)

    print(files_name_cc100[0])

    ds = tf.data.TextLineDataset(
       files_name_cc100
    )
    ds = ds.map(lambda *ex: dict(zip(['title', 'text'], ['None',ex[0]])))
    ds = ds.shuffle(buffer_size=1000000)

    return ds

MODEL_SIZE = 'large'

vocab = "gs://translationv2/models/viT5_1024_{MODEL_SIZE}/wiki_vietnamese_vocab.model"
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


model_parallelism, train_batch_size, keep_checkpoint_max = {
    'small': (1, 256, 16),
    'base': (2, 128, 8),
    'large': (8, 128, 4),
    '3B': (8, 16, 1),
    '11B': (8, 16, 1),
}[MODEL_SIZE]

model_dir = f'gs://translationv2/models/viT5_1024_{MODEL_SIZE}'

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