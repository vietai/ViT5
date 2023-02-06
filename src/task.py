import seqio
import t5
import tensorflow as tf
import functools
from typing import Dict
import metrics
import os

TaskRegistry = seqio.TaskRegistry

DEFAULT_ViT5_SPM_PATH = "gs://vietai_public/viT5/vocab/spiece.model"
DEFAULT_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(
        vocabulary=seqio.SentencePieceVocabulary(DEFAULT_ViT5_SPM_PATH), add_eos=True,
        required=False),
    "targets": seqio.Feature(
        vocabulary=seqio.SentencePieceVocabulary(DEFAULT_ViT5_SPM_PATH), add_eos=True)
}

DEFAULT_MT5_SPM_PATH = "gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model"  # GCS
MT5_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(
        vocabulary=seqio.SentencePieceVocabulary(DEFAULT_MT5_SPM_PATH), add_eos=True,
        required=False),
    "targets": seqio.Feature(
        vocabulary=seqio.SentencePieceVocabulary(DEFAULT_MT5_SPM_PATH), add_eos=True)
}


def registerTask(task: str, splits: Dict, metric_name: str=None, vocab: str='default'):
  vocab_map = {
    'default': DEFAULT_OUTPUT_FEATURES,
    'mt5': MT5_OUTPUT_FEATURES
  }

  def parseDataset(split: str, shuffle_files: bool = False, seed: int = 0):
    ds = tf.data.TextLineDataset([splits[split]])
    ds = ds.map(
    functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                      field_delim="\t", use_quote_delim=False),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(lambda *ex: dict(zip(["input", "target"], ex)))
    return ds
  def preprocessor(ds):
    def to_inputs_and_targets(ex):
        return {
            "inputs":ex["input"],
            "targets": ex["target"]
        }
    return ds.map(to_inputs_and_targets, 
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
  
  TaskRegistry.remove(task)
  TaskRegistry.add(
      task,
      source=seqio.FunctionDataSource(
          dataset_fn=parseDataset,
          splits=list(splits.keys())
      ),
      preprocessors=[preprocessor, 
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      output_features=vocab_map[vocab],
      metric_fns=[metrics.map_name_to_metric_function(metric_name)] if metric_name else []
  )
