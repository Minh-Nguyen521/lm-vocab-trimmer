import logging
import os

os.environ["HF_DATASETS_CACHE"] = os.path.expanduser("~/.cache/huggingface/datasets")

from vocabtrimmer import VocabTrimmer

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

MODEL = "google/embeddinggemma-300m"
LANGUAGE = "vi"
PATH_TO_SAVE = "model/gemma300-vi-trimmed"
DATASET_FILES = [
    "crosslingual/original/merged_queries_vi.json",
    "crosslingual/eval/filtered_corpus.json",
]

trimmer = VocabTrimmer(MODEL)
trimmer.show_parameter()
trimmer.trim_vocab(
    language=LANGUAGE, path_to_save=PATH_TO_SAVE, dataset=DATASET_FILES, min_frequency=1
)
trimmer.show_parameter()
