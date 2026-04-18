import logging

from vocabtrimmer import VocabTrimmer

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

PATH_TO_SAVE = "model/gemma300-vi-legal-trimmed-crosslingual"

trimmer = VocabTrimmer("google/embeddinggemma-300m")
trimmer.show_parameter()
trimmer.trim_vocab(
    language="vi",
    path_to_save=PATH_TO_SAVE,
    dataset="bkai-foundation-models/crosslingual",
    dataset_column="pos",
    dataset_split="train",
    dataset_data_files=["synthetic/cross_corpus.parquet", "synthetic/cross_queries.parquet", "original/merged_queries_vi.json"],
    min_frequency=1,
    streaming=True,
)
trimmer.show_parameter()
