from click import style
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
parser.add_argument('-model', dest='model', type=str, help='model dir', default='.')

args = parser.parse_args()

TPU_TOPOLOGY = 'v3-8'
TPU_ADDRESS = args.tpu
TPU_ADDRESS = f'grpc://{TPU_ADDRESS}:8470'

print(f"TPU Address {TPU_ADDRESS}")
ON_CLOUD = True


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

MODEL_SIZE = "base"

# Set parallelism and batch size to fit on v3-8 TPU (if possible).
model_parallelism, train_batch_size, keep_checkpoint_max = {
    "small": (1, 256, 16),
    "base": (4, 256, 8),
    "large": (8, 256, 4),
    "3B": (8, 16, 1),
    "11B": (8, 16, 1)}[MODEL_SIZE]

MODEL_DIR = args.model
model = MtfModel(
    model_dir=MODEL_DIR,
    tpu=TPU_ADDRESS,
    tpu_topology=TPU_TOPOLOGY,
    model_parallelism=model_parallelism,
    batch_size=train_batch_size,
    sequence_length={"inputs": 512, "targets": 512},
    learning_rate_schedule=0.001,
    save_checkpoints_steps=0,
    keep_checkpoint_max=keep_checkpoint_max if ON_CLOUD else None,
    # iterations_per_loop=100,
)

import re
import os.path

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# Manually apply preprocessing by prepending "triviaqa question:".
vocab = f"gs://translationv2/models/spm/cc100_envi_vocab.model"

input_files = os.listdir('en')
input_files.sort(key=natural_keys)
input_files = input_files[::-1]

existed_file = list(map(lambda x: x.split('-')[0], os.listdir('vi')))


for input_file in input_files:
    # Ignore any logging so that we only see the model's answers to the questions.
    output_file = input_file.replace('en', 'vi')
    import time
    start_time = time.time()
    print('Starting ', input_file)
    predict_inputs_path = f'en/{input_file}'
    predict_outputs_path = f"vi/{output_file}"

    
    if output_file in existed_file:
        print('skipping file ', predict_outputs_path)
        continue
    with tf_verbosity_level('ERROR'):
        model.batch_size = 8  # Min size for small model on v2-8 with parallelism 1.
        model.predict(
            input_file=predict_inputs_path,
            output_file=predict_outputs_path,
            # Select the most probable output token at each step.
            vocabulary=t5.data.SentencePieceVocabulary(vocab),
            checkpoint_steps=-1,
            temperature=0,
        )
    print('End ', input_file, )
    print("--- %s seconds ---" % (time.time() - start_time))

