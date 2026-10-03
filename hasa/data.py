"""Article-title next-word prediction benchmark (paper Sec. V-A).

Medium article titles, one client per publication (K = 7). The order of the
random-number calls follows the original experiment code exactly, so the split
(and therefore the heterogeneity scores and subnet widths) is identical to the
one used in the paper.
"""
import re
import unicodedata
import warnings
from collections import Counter
from types import SimpleNamespace

import demoji
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from bs4 import BeautifulSoup
from torch.utils.data import DataLoader, TensorDataset

from .utils import set_all_seeds

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------- text cleaning
def remove_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()


def remove_non_alphabetic(text):
    return re.sub("[^a-zA-Z]", " ", text)


def remove_emojis(text):
    return demoji.replace(text, "")


def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = remove_html_tags(text)
    text = remove_emojis(text)
    text = remove_non_alphabetic(text)
    text = text.lower()
    return " ".join(text.split())


class BasicEnglishTokenizer:
    """Lowercase + whitespace split."""

    def __call__(self, text: str):
        return unicodedata.normalize("NFKD", text).lower().split()


class Vocabulary:
    """Token-to-index vocabulary sorted by frequency, then alphabetically."""

    def __init__(self, tokens, min_freq=1, specials=None):
        self.token2idx, self.idx2token = {}, []
        freqs = Counter()
        for token_list in tokens:
            freqs.update(token_list)
        for tok in specials or []:
            self.idx2token.append(tok)
            self.token2idx[tok] = len(self.token2idx)
        for tok, freq in sorted(freqs.items(), key=lambda x: (-x[1], x[0])):
            if freq >= min_freq and tok not in self.token2idx:
                self.idx2token.append(tok)
                self.token2idx[tok] = len(self.token2idx)

    def __len__(self):
        return len(self.idx2token)

    def __getitem__(self, token):
        return self.token2idx.get(token, self.token2idx["<oov>"])

    def __contains__(self, token):
        return token in self.token2idx


# ---------------------------------------------------------------- sequences
def make_ngrams(tokens):
    """All prefixes of length >= 2: the last token is the next-word target."""
    return [tokens[: i + 1] for i in range(1, len(tokens))]


def to_ids(ngram, features_vocab, target_vocab):
    input_ids = [features_vocab[tok] for tok in ngram[:-1]]
    return input_ids, target_vocab[ngram[-1]]


def prepare_dataset(sequences, max_len):
    X, y = [], []
    for seq in sequences:
        input_ids, target_id = seq[0], seq[1]
        X.append(F.pad(torch.tensor(input_ids), (max_len - len(input_ids), 0), value=0))
        y.append(target_id)
    return TensorDataset(torch.stack(X), torch.tensor(y))


def partition_by_publication(sequences, df, num_clients, global_test_frac=0.05):
    """One client per publication (largest first); 5% of each client's articles go to the
    global in-federation test set. Clients never share articles."""
    article_to_sequences = {}
    for input_ids, target_id, article_id in sequences:
        article_to_sequences.setdefault(article_id, []).append((input_ids, target_id))

    pub_counts = {}
    for article_id, seqs in article_to_sequences.items():
        pub = df.iloc[article_id]["publication"]
        pub_counts.setdefault(pub, {"articles": [], "sequences": 0})
        pub_counts[pub]["articles"].append(article_id)
        pub_counts[pub]["sequences"] += len(seqs)

    sorted_pubs = sorted(pub_counts.items(), key=lambda x: x[1]["sequences"], reverse=True)
    top_pubs = [p for p, _ in sorted_pubs[: min(num_clients, len(sorted_pubs))]]

    global_infed_test_ids = set()
    client_article_ids = []
    for pub in top_pubs:
        articles = pub_counts[pub]["articles"]
        np.random.shuffle(articles)
        n_global = max(1, int(len(articles) * global_test_frac))
        global_infed_test_ids.update(articles[:n_global])
        client_article_ids.append(articles[n_global:])
    return client_article_ids, top_pubs, global_infed_test_ids


# ---------------------------------------------------------------- benchmark
def load_benchmark(data_path="data/medium_data.csv", num_clients=7, batch_size=64, seed=42, verbose=True):
    """Build the full benchmark: vocabularies, 7 clients with 70/10/20 article-level
    train/val/test splits, and the two global test sets."""
    say = print if verbose else (lambda *a, **k: None)
    try:
        demoji.download_codes()
    except Exception:
        pass

    df = pd.read_csv(data_path)
    df["cleaned_title"] = df["title"].apply(preprocess_text)
    df = df[df["cleaned_title"].str.split().str.len() >= 3].reset_index(drop=True)
    tokenizer = BasicEnglishTokenizer()
    df["tokens"] = df["cleaned_title"].apply(tokenizer)

    set_all_seeds(seed)
    all_ids = list(range(len(df)))
    np.random.shuffle(all_ids)
    n_ood = int(0.05 * len(all_ids))
    global_ood_ids = set(all_ids[:n_ood])
    training_ids = set(all_ids[n_ood:])

    # vocabularies from the training pool only (no test leakage)
    pool_titles = [df.iloc[i]["tokens"] for i in training_ids]
    features_vocab = Vocabulary(pool_titles, min_freq=2, specials=["<pad>", "<oov>"])
    target_vocab = Vocabulary(pool_titles, min_freq=2, specials=["<oov>"])

    all_tokens = df["tokens"].tolist()

    def sequences_of(article_ids):
        out = []
        for aid in article_ids:
            for ng in make_ngrams(all_tokens[aid]):
                out.append((*to_ids(ng, features_vocab, target_vocab), aid))
        return out

    training_sequences = sequences_of(training_ids)
    global_ood_sequences = sequences_of(global_ood_ids)
    max_seq_length = max(len(s[0]) for s in training_sequences)

    client_article_ids, client_names, global_infed_ids = partition_by_publication(
        training_sequences, df, num_clients
    )
    global_infed_sequences = sequences_of(global_infed_ids)

    clients = []
    for article_ids in client_article_ids:
        ids = np.array(article_ids)
        np.random.shuffle(ids)
        n_train, n_val = int(0.7 * len(ids)), int(0.1 * len(ids))
        train_ids = set(ids[:n_train])
        val_ids = set(ids[n_train:n_train + n_val])
        test_ids = set(ids[n_train + n_val:])
        split = {"train": [], "val": [], "test": []}
        for input_ids, target_id, aid in training_sequences:
            if aid in train_ids:
                split["train"].append((input_ids, target_id))
            elif aid in val_ids:
                split["val"].append((input_ids, target_id))
            elif aid in test_ids:
                split["test"].append((input_ids, target_id))
        ds = {k: prepare_dataset(v, max_seq_length) for k, v in split.items()}
        clients.append(SimpleNamespace(
            train_article_ids=train_ids,
            n_articles=len(article_ids),
            n_train_articles=len(train_ids), n_val_articles=len(val_ids), n_test_articles=len(test_ids),
            sizes={k: len(v) for k, v in ds.items()},
            loaders={
                "train": DataLoader(ds["train"], batch_size=batch_size, shuffle=True),
                "val": DataLoader(ds["val"], batch_size=batch_size, shuffle=False),
                "test": DataLoader(ds["test"], batch_size=batch_size, shuffle=False),
            },
        ))

    bench = SimpleNamespace(
        df=df, tokenizer=tokenizer,
        features_vocab=features_vocab, target_vocab=target_vocab, max_seq_length=max_seq_length,
        client_names=client_names, clients=clients, num_clients=len(clients),
        client_data_sizes=[c.sizes["train"] for c in clients],
        global_ood_loader=DataLoader(prepare_dataset(global_ood_sequences, max_seq_length), batch_size=batch_size, shuffle=False),
        global_infed_loader=DataLoader(prepare_dataset(global_infed_sequences, max_seq_length), batch_size=batch_size, shuffle=False),
        n_global_ood_articles=len(global_ood_ids), n_global_ood_sequences=len(global_ood_sequences),
        n_global_infed_articles=len(global_infed_ids), n_global_infed_sequences=len(global_infed_sequences),
    )
    say(f"Articles after cleaning: {len(df)} | vocabulary: {len(features_vocab)} inputs, {len(target_vocab)} targets")
    say(f"Global OOD test: {bench.n_global_ood_articles} articles, {bench.n_global_ood_sequences} sequences | "
        f"global in-federation test: {bench.n_global_infed_sequences} sequences")
    for name, c in zip(client_names, clients):
        say(f"  {name:24s} {c.n_articles:5d} articles -> train/val/test sequences "
            f"{c.sizes['train']:6d} / {c.sizes['val']:5d} / {c.sizes['test']:5d}")
    return bench
