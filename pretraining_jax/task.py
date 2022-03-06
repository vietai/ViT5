import seqio
import t5
from random import shuffle
import tensorflow as tf
from t5.data import preprocessors

TaskRegistry = seqio.TaskRegistry

vocab = f"gs://translationv2/models/spm/cc100_envi_vocab.model"

DEFAULT_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(
        vocabulary=t5.data.SentencePieceVocabulary(vocab), add_eos=True,
        required=False),
    "targets": seqio.Feature(
        vocabulary=t5.data.SentencePieceVocabulary(vocab), add_eos=True)
}


def dumping_dataset(split, shuffle_files = False):
    del shuffle_files
    files_name_cc100 = [f'gs://translationv2/data/cc100_envi_1024/train_envi_{i}.txt' for i in range(0,310)]

    shuffle(files_name_cc100)

    print(files_name_cc100[0])

    ds = tf.data.TextLineDataset(
       files_name_cc100
    )
    ds = ds.map(lambda *ex: dict(zip(['title', 'text'], ['None',ex[0]])))
    ds = ds.shuffle(buffer_size=1000000)

    return ds

TaskRegistry.remove('dumping_dataset')
TaskRegistry.add(
    "dumping_dataset",
    source=dumping_dataset,
    preprocessors=[
        functools.partial(
            preprocessors.rekey, key_map={
                "inputs": None,
                "targets": "text"
            }),
        seqio.preprocessors.tokenize,
        # seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,

    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[])