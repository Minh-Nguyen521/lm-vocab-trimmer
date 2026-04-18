import logging

from vocabtrimmer import VocabTrimmer

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

PATH_TO_SAVE = "model/gemma300-vi-legal-trimmed"

trimmer = VocabTrimmer("google/embeddinggemma-300m")
trimmer.show_parameter()
trimmer.trim_vocab(
    language="vi",
    path_to_save=PATH_TO_SAVE,
    dataset="phamson02/large-vi-legal-queries",
    dataset_column="context",
    dataset_split="train",
    min_frequency=1,
    streaming=True,
)
trimmer.show_parameter()
