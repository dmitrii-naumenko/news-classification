from typing import Dict, List
import multiprocessing

import itertools
import numpy as np
from collections import Counter
from joblib import delayed
from joblib import Parallel

from ml.preprocess import Preprocessor
from ml.classify_job import job_batch


class DuplicatesProcessor:
    """Class for deleting duplicated texts from a corpus."""
    def __init__(self,
                 max_len: int = 40,
                 threshold: float = 0.8,
                 min_token_occurancies: int = 1):

        self.max_len = max_len
        self.min_token_occurancies = min_token_occurancies
        self.threshold = threshold
        self.prepr = Preprocessor()
        pass

    def drop_duplicated(self, corpus: np.ndarray) -> np.ndarray:
        all_tokens = self.get_all_tokens(corpus, self.min_token_occurancies)
        vocab = self.create_vocab(all_tokens)
        idxs_text = self._corpus_to_token_idxs(corpus, vocab, self.max_len)
        idxs_text = [set(x) for x in idxs_text]

        pairs = []
        min_length = 5

        n_cores = multiprocessing.cpu_count()
        #print('n_cores =', n_cores)
        batch_size = len(corpus) // (n_cores * 5)
        batch_size = 1 if batch_size < 1 else batch_size

        # backend="multiprocessing"
        pairs_list_of_list = Parallel(n_jobs=-1, backend="threading")\
               (delayed(job_batch)(i, batch_size, idxs_text, min_length, self.threshold) 
                       for i in range(0, len(corpus)-1, batch_size))
        pairs = [item for sublist in pairs_list_of_list for item in sublist]

        pairs = self.extend_matches(pairs)
        for p in pairs:
            best_ind = p[np.argmax(np.array([len(x) for x in corpus[p]]))]
            best_sentence = corpus[best_ind]
            for _ in p:
                corpus[p] = best_sentence

        return np.unique(corpus)

    def extend_matches(self, groups: List[List[int]]) -> List[List[int]]:
        sets = []
        for g in groups:
            g = set(g)
            skip_adding = False
            for i in range(len(sets)):
                s = sets[i]
                if len(g & s) != 0:
                    sets[i] = s | g
                    skip_adding = True
                    break

            if not skip_adding:
                sets.append(g)

        return [list(x) for x in sets]

    def _corpus_to_token_idxs(self, corpus, vocab, max_len) -> List[List[int]]:
        tokenized_text = list(map(self.prepr.simple_preproc, corpus))
        token_idxs_text = [[vocab.get(x, vocab['<unk>']) for x in text[:max_len]] for text in tokenized_text]
        return token_idxs_text

    def create_vocab(self, inner_keys: List[str]) -> Dict[str, int]:
        inner_keys = ['<pad>', '<unk>'] + inner_keys
        vocab = {}
        for idx, word in enumerate(inner_keys):
            vocab[word] = idx
        return vocab

    def get_all_tokens(self, corpus: np.ndarray,
                       min_occurancies: int) -> List[str]:
        sentences = np.unique(corpus)
        tokens = list(itertools.chain.from_iterable(map(self.prepr.simple_preproc, sentences)))
        tokens = self.prepr._filter_rare_words(Counter(tokens), min_occurancies)

        return tokens
