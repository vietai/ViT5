import sklearn.metrics
import numpy as np
from scipy.stats import pearsonr as scipy_pearsonr
import re
from datasets import load_metric

def map_name_to_metric_function(name):
  dict_ = {
    "rouge": rouge,
    "macro_f1": macro_f1,
    "accuracy": accuracy,
  }
  return dict_[name]


def rouge(targets, predictions):
  rouge = load_metric('rouge')
  results = rouge.compute(predictions=predictions,references=targets)
  results = {k:v.mid.fmeasure for k,v in results.items()}
  return results

def accuracy(targets, predictions):
  return {"accuracy": 100*sklearn.metrics.accuracy_score(targets, predictions)}

def macro_f1(targets, predictions):
  return {f"f1_macro": 100*sklearn.metrics.f1_score(targets, predictions, average="macro")}



