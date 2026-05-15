"""Simple whitespace-based tokenizer / vectorizer for short social-media text.

This minimal utility builds a token->index vocabulary from a list of texts,
supports encoding to padded integer tensors, and saving/loading the vocab.

Design choices:
- `0` is the padding index, `1` is the unknown token index.
- Tokenization is simple whitespace lowercased split; this is sufficient for
  a quick baseline and keeps the P7 implementation lightweight.
"""

from collections import Counter
import json
from pathlib import Path
from typing import Iterable, List

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class TextVectorizer:
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self._frozen = False

    def fit(self, texts: Iterable[str], max_vocab: int = 20000) -> None:
        if self._frozen:
            return
        counter = Counter()
        for t in texts:
            if not t:
                continue
            toks = t.lower().split()
            counter.update(toks)

        # Reserve 0 and 1 for PAD and UNK
        idx = 2
        for token, freq in counter.most_common(max_vocab):
            if freq < self.min_freq:
                break
            if token in self.vocab:
                continue
            self.vocab[token] = idx
            idx += 1

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, texts: List[str], max_len: int = 64) -> (List[List[int]], List[int]):
        ids_batch = []
        lengths = []
        for t in texts:
            toks = t.lower().split() if t else []
            ids = [self.vocab.get(tok, self.vocab[UNK_TOKEN]) for tok in toks][:max_len]
            lengths.append(len(ids))
            # pad
            if len(ids) < max_len:
                ids = ids + [self.vocab[PAD_TOKEN]] * (max_len - len(ids))
            ids_batch.append(ids)
        return ids_batch, lengths

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)

    def load(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
