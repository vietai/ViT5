import tensorflow
import functools
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import argparse
import tensorflow.compat.v1 as tf
import t5
from t5.models import MtfModel
import os
from datasets import load_metric

print(tensorflow.__version__)


parser = argparse.ArgumentParser(description='Finetunning')
parser.add_argument('-tpu', dest='tpu', type=str, help='tpu address', default='0.0.0.0')
parser.add_argument('-steps', dest='steps', type=int, help='finetune steps', default=100000)
parser.add_argument('-model_size', dest='model_size', type=str, help='Model Size', default='base')
parser.add_argument('-pretraining_path', dest='pretraining_path', type=str, help='Pretraining Path', required=True)
parser.add_argument('-output_path', dest='output_path', type=str, help='Output Path', required=True)

args = parser.parse_args()

TPU_TOPOLOGY = 'v2-8'
TPU_ADDRESS = args.tpu
TPU_ADDRESS = f'grpc://{TPU_ADDRESS}:8470'

MODEL_SIZE = args.model_size


print(f"TPU Address {TPU_ADDRESS}")
print(f"FINE TUNE STEPS {args.steps}")
ON_CLOUD = True

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
# set to True to use public checkpoint, else for internal vietAI
vietai_public_checkpoint = False
if vietai_public_checkpoint:
  BASE_DIR = "gs://vietai_public/viT5"
else:
  BASE_DIR = 'gs://translationv2/models'


task = "sum"
vocab = f"gs://vietai_public/viT5/viT5_base_1024/spiece.model"
def dumping_dataset(split, shuffle_files = False):
    del shuffle_files
    if split == 'train':
      ds = tf.data.TextLineDataset(
            [
            f'gs://vietai_public/viT5/data/vietnews/train_dedup.tsv',
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
    ds = ds.shuffle(buffer_size=10000)
    ds = ds.map(lambda *ex: dict(zip(["input", "target"], ex)))
    return ds

def faq_preprocessor(ds):

  def to_inputs_and_targets(ex):
    """Map {"inputs": ..., "targets": ...}->{"inputs": ner..., "targets": ...}."""
    return {
        "inputs":
            tf.strings.join(
                [ex["input"]]),
        "targets": ex["target"]
    }
  return ds.map(to_inputs_and_targets, 
                num_parallel_calls=tf.data.experimental.AUTOTUNE)


t5.data.TaskRegistry.remove(task)
t5.data.TaskRegistry.add(
    task,
    dataset_fn=dumping_dataset,
    splits=["train", "validation"],
    text_preprocessor=[faq_preprocessor],
    metric_fns=[],
    output_features=t5.data.Feature(vocabulary=t5.data.SentencePieceVocabulary(vocab))
)

t5.data.MixtureRegistry.remove("all")
t5.data.MixtureRegistry.add(
    "all",
    [task],
     default_rate=1.0,
)

# Set parallelism and batch size to fit on v3-8 TPU (if possible).
model_parallelism, train_batch_size, keep_checkpoint_max = {
    "small": (1, 256, 16),
    "base": (4, 256, 8),
    "large": (8, 256, 4),
    "3B": (8, 16, 1),
    "11B": (8, 16, 1)}[MODEL_SIZE]

PRETRAINED_DIR = args.pretraining_path


# change to your own MODEL_DIR 
MODEL_DIR = args.output_path
tf.io.gfile.makedirs(MODEL_DIR)
# The models from paper are based on the Mesh Tensorflow Transformer.

model = MtfModel(
    model_dir=MODEL_DIR,
    tpu=TPU_ADDRESS,
    tpu_topology=TPU_TOPOLOGY,
    model_parallelism=model_parallelism,
    batch_size=train_batch_size,
    sequence_length={"inputs": 1024, "targets": 256},
    learning_rate_schedule=0.001,
    iterations_per_loop=100,
    save_checkpoints_steps=20000,
    keep_checkpoint_max=10,
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


with open(f'../data/{task}/test.tsv') as file:
  with open('predict_input.txt', 'w') as file_p:
    with open('actual_output.txt', 'w') as file_a:
      for line in file:
        line = line.strip().split('\t')
        file_p.write(f'{line[0]}\n')
        file_a.write(f'{line[0]}\n')

for t in tasks:
  dir = t[0]
  task = t[1]

  
  input_file = 'predict_input.txt'
  output_file = 'predict_output.txt'

  # Write out the supplied questions to text files.
      

  predict_inputs_path = input_file
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