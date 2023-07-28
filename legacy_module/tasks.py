import json
import functools
import t5.data
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import seqio
import logging
import os
logging.basicConfig(level=logging.INFO)

TaskRegistry = seqio.TaskRegistry

PRETRAINED_VOCAB_PATH = "gs://bt5-small-bucket/new_vocabs/50k/new_vocab.model"
PRETRAINING_DATA_PATH = "gs://bt5-small-bucket/lsh_deduplicated_data/*.txt"

def get_line_count(pattern, split="train"):
    input_files = tf.io.gfile.glob(pattern)
    count_file = os.path.join(os.path.dirname(pattern), "samples.count")
    line_count = 0

    if tf.io.gfile.exists(count_file):
        line_count = json.load(tf.io.gfile.GFile(count_file))
    else:
        for input_file in input_files:
            with tf.io.gfile.GFile(input_file) as inpf:
                for _ in inpf: line_count += 1

        json.dump(line_count, tf.io.gfile.GFile(count_file, 'w'))

    return {split: line_count}

def get_vocabulary():
    return seqio.SentencePieceVocabulary(
        PRETRAINED_VOCAB_PATH
    )

OUTPUT_FEATURES = {
    "targets": seqio.Feature(
        vocabulary=get_vocabulary(), add_eos=True)
}

TaskRegistry.add(
    "blamda",
    source=seqio.TextLineDataSource(
        split_to_filepattern={"train": PRETRAINING_DATA_PATH},
        num_input_examples=get_line_count(PRETRAINING_DATA_PATH)
    ),
    preprocessors=[
        functools.partial(
            t5.data.preprocessors.parse_tsv,
            field_names=["text"]),
        functools.partial(
            t5.data.preprocessors.rekey,
            key_map={
                "inputs": None,
                "targets": "text"
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        t5.data.preprocessors.full_lm,
        # seqio.preprocessors.append_eos_after_trim
    ],
    output_features=OUTPUT_FEATURES,
    metric_fns=[])

TaskRegistry.add(
    "bt5",
    source=seqio.TextLineDataSource(
        split_to_filepattern={"train": PRETRAINING_DATA_PATH},
        num_input_examples=get_line_count(PRETRAINING_DATA_PATH)
    ),
    preprocessors=[
        functools.partial(
            t5.data.preprocessors.parse_tsv,
            field_names=["text"]),
        functools.partial(
            t5.data.preprocessors.rekey,
            key_map={
                "inputs": None,
                "targets": "text"
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        t5.data.preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim
    ],
    output_features=OUTPUT_FEATURES,
    metric_fns=[])

seqio.MixtureRegistry.add("blamda", ["blamda"], default_rate=1.0)
seqio.MixtureRegistry.add("bt5", ["bt5"], default_rate=1.0)

if __name__ == "__main__":
    task = seqio.TaskRegistry.get("blamda")
    ds = task.get_dataset(split="train", sequence_length={"targets": 128})
    for ex in tfds.as_numpy(ds.take(5)):
        print(ex["targets"].shape)


