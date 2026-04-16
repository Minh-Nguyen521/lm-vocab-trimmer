import logging
import os

from vocabtrimmer import VocabTrimmer

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

LANGUAGE = "vi"
PATH_TO_SAVE = "model/gemma300-vi-trimmed"
DATASET_FILES = [
    "crosslingual/original/merged_queries_vi.json",
    "crosslingual/eval/filtered_corpus.json",
    # "crosslingual/synthetic/cross_corpus.parquet",
    # "crosslingual/synthetic/cross_queries.parquet"
]

trimmer = VocabTrimmer("google/embeddinggemma-300m")
trimmer.show_parameter()
trimmer.trim_vocab(
    language="vi", path_to_save=PATH_TO_SAVE, dataset=DATASET_FILES, min_frequency=1
)
trimmer.show_parameter()
