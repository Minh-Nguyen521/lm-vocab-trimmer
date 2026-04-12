"""Mining language specific vocabulary from corpus"""

import json
import logging
import os
from collections import defaultdict
from itertools import chain

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from .util import DEFAULT_CACHE_DIR

__all__ = "vocab_miner"


def update_fq(tokens, fq):
    for w in tokens:
        fq[w] += 1
    return fq


def _load_local(path, dataset_column):
    """Load texts from a local JSON or parquet file."""
    if path.endswith(".json"):
        with open(path) as f:
            raw = json.load(f)
        texts = list(raw.values()) if isinstance(raw, dict) else [t[dataset_column] for t in raw]
    elif path.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(path)
        texts = df[dataset_column].dropna().tolist()
        texts = [t if isinstance(t, str) else " ".join(t) for t in texts]
    else:
        raise ValueError(f"Unsupported local file format: {path}")
    return [{"text": t} for t in texts], "text"


def vocab_miner(
    model: str = "google/mt5-small",
    language: str = "ja",
    dataset: str = "mc4",
    dataset_column: str = "text",
    dataset_name: str = None,
    dataset_split: str = "train",
    target_vocab_size: int = None,
    min_frequency: int = 2,
    chunk: int = 1000,
    cache_file_vocab: str = None,
    cache_file_frequency: str = None,
    overwrite: bool = False,
):
    """Mining language specific vocabulary from corpus

    :param model: model name on huggingface or path to local model
    :param language: language code of tokens to keep
    :param dataset: huggingface dataset name, local file path (.json/.parquet), or list of local paths
    :param dataset_column: column of the dataset containing text for mining
    :param dataset_name: name of the dataset
    :param dataset_split: split of the dataset
    :param target_vocab_size: vocab size after mining
    :param min_frequency: min frequency of tokens
    :param chunk: chunk size at mining
    :param cache_file_vocab: cache directly to save the mined vocab
    :param cache_file_frequency: cache directly to save the frequency over the corpus used for vocab mining
    :return: a dictionary of {token: token_id}
    """

    dataset_name = (
        language
        if dataset in ["mc4", "vocabtrimmer/mc4_validation"] and dataset_name is None
        else dataset_name
    )
    logging.info(
        f"[DATASET INFO] dataset: {dataset}, name: {dataset_name}, split: {dataset_split}, column: {dataset_column}"
    )
    logging.info(f"[MINING INFO] language: {language}, model: {model}, chunk: {chunk}")

    dataset_key = "_".join(os.path.basename(d) for d in dataset) if isinstance(dataset, list) else dataset
    if cache_file_frequency is None:
        cache_file_frequency = f"{DEFAULT_CACHE_DIR}/vocab_mining/frequency.{dataset_key}.{dataset_column}.{dataset_name}.{dataset_split}.{model.replace('/', '_')}.json"
    if cache_file_vocab is None:
        cache_file_vocab = f"{DEFAULT_CACHE_DIR}/vocab_mining/vocab.{dataset_key}.{dataset_column}.{dataset_name}.{dataset_split}.{model.replace('/', '_')}.{target_vocab_size}.{min_frequency}.json"

    if not overwrite and os.path.exists(cache_file_vocab):
        logging.info(f"load vocab file from {cache_file_vocab}")
        with open(cache_file_vocab) as f:
            return json.load(f)

    os.makedirs(os.path.dirname(cache_file_frequency), exist_ok=True)
    os.makedirs(os.path.dirname(cache_file_vocab), exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model)
    if not os.path.exists(cache_file_frequency):
        os.makedirs(os.path.dirname(cache_file_frequency), exist_ok=True)

        # processing dataset — supports single HF dataset, single local file, or list of local files
        datasets = dataset if isinstance(dataset, list) else [dataset]
        fq = defaultdict(int)
        for ds in datasets:
            if os.path.exists(ds):
                ds, col = _load_local(ds, dataset_column)
            else:
                ds = load_dataset(ds, dataset_name, split=dataset_split)
                col = dataset_column

            logging.info(f"caching all tokens to {cache_file_frequency}")
            batch = []
            for t in tqdm(ds):
                batch.append(t[col])
                if len(batch) >= chunk:
                    fq = update_fq(chain(*tokenizer(batch, truncation=True, max_length=tokenizer.model_max_length)["input_ids"]), fq)
                    batch = []
            if len(batch) != 0:
                fq = update_fq(chain(*tokenizer(batch, truncation=True, max_length=tokenizer.model_max_length)["input_ids"]), fq)

        logging.info(f"saving frequency file to {cache_file_frequency}")
        with open(cache_file_frequency, "w") as f:
            json.dump(fq, f)

    logging.info(f"load frequency file from {cache_file_frequency}")
    with open(cache_file_frequency) as f:
        freq = [
            (tokenizer.convert_ids_to_tokens(int(k)), v, int(k))
            for k, v in json.load(f).items()
            if v >= min_frequency
        ]

    freq = sorted(freq, key=lambda x: x[1], reverse=True)
    if target_vocab_size is not None:
        assert target_vocab_size < len(freq), (
            "vocabulary size is already smaller than the target_vocab_size"
        )
        freq = freq[:target_vocab_size]
    new_vocab = {x[0]: x[2] for x in freq}
    logging.info(f"save vocab file at {cache_file_vocab}")
    with open(cache_file_vocab, "w") as f:
        json.dump(new_vocab, f)
    return new_vocab
