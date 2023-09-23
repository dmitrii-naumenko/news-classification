import string
from collections import Counter
from typing import Dict, List, Tuple, Union, Callable

import itertools
import re
import math
import nltk
import numpy as np
import pandas as pd
import torch
from navec import Navec
import yaml

from ml.preprocess import Preprocessor

class ClassifyProcessor:
    """Class for classifying texts from a corpus."""
    def __init__(self,
                max_len: int = 40,
                min_token_occurancies: int = 1,
                emb_rand_uni_bound: float = 0.2,
                random_seed: int = 0,
                navec_path: str = 'model/navec_hudlit_v1_12B_500K_300d_100q.tar'):

        self.random_seed = random_seed
        self.max_len = max_len
        self.min_token_occurancies = min_token_occurancies
        self.emb_rand_uni_bound = emb_rand_uni_bound
        with open('model/categories.yml', 'r') as f:
            self.categories = yaml.safe_load(f)

        self.prepr = Preprocessor()
        self.navec = Navec.load(navec_path)   
  

    def classify(self, corpus: np.ndarray):

        all_tokens, unk_keywords = self.get_all_tokens(corpus, self.min_token_occurancies)

        emb_matrix, vocab, unk_words = self.create_navec_emb(
            all_tokens, self.random_seed, self.emb_rand_uni_bound)

        embeddings_layer = torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(emb_matrix),
            freeze=True,
            padding_idx=0
        )

        # shape = [Batch, Left]
        doc, doc_length, tokenized_doc = self._corpus_to_token_idxs(corpus, vocab, padding=False)

        queries, _, _ = self._corpus_to_token_idxs([' '.join(x[1]) for x in self.categories], vocab)
        queries = torch.LongTensor(queries)
        result = np.zeros((len(doc), len(queries)))

        for i, d in enumerate(doc):
            # shape = [Batch, Right]
            dd = torch.LongTensor([d for _ in range(len(queries))])

            # shape = [Batch, Left, Right]
            matching_matrix = self._get_matching_matrix(embeddings_layer, queries, dd)
            matching_matrix = torch.pow(matching_matrix, 3)
            y = matching_matrix.sum(axis=1).sum(axis=1) / torch.LongTensor([torch.count_nonzero(q) for q in queries])
            result[i,:] = y

        predicted = [self.categories[x][0] for x in result.argmax(axis=1)]

        for i in range(len(predicted)):
            s = corpus[i].lower()
            if doc_length[i] <= 6:
                predicted[i] = 'Другое'
            elif 'всу' in tokenized_doc[i]:
                predicted[i] = 'Политика'
            elif 'криптовалюта' in s or 'биткоин' in s or 'блокчейн' in s :
                predicted[i] = 'Криптовалюты' 

        df = pd.DataFrame()
        df['text'] = corpus
        df['category'] = predicted

        return df


    def _corpus_to_token_idxs(self, corpus, vocab, padding=True) -> List[List[int]]:
        tokenized_text = list(map(self.prepr.simple_preproc, corpus))
        token_idxs_text = [[vocab.get(x, vocab['<unk>']) for x in text[:self.max_len]] for text in tokenized_text]
        length = [len(x) for x in token_idxs_text]

        if padding:
            max_text_len = max(length)
            token_idxs_text = [x + [0] * (max_text_len-len(x)) for x in token_idxs_text]

        return token_idxs_text, length, tokenized_text     

    def _get_matching_matrix(self, embeddings, query: torch.Tensor, doc: torch.Tensor) -> torch.FloatTensor:
        # shape = [B, L, D]
        embed_query = embeddings(query.long())
        # shape = [B, R, D]
        embed_doc = embeddings(doc.long())

        # shape = [B, L, R]
        matching_matrix = torch.einsum(
            'bld,brd->blr',
            torch.nn.functional.normalize(embed_query, p=2, dim=-1),
            torch.nn.functional.normalize(embed_doc, p=2, dim=-1)
        )
        return matching_matrix       


    def get_all_tokens(self, corpus: np.ndarray, 
                       min_occurancies: int) -> Tuple[List[str], List[str]]:
        sentences = np.unique(corpus)
        tokens = list(itertools.chain.from_iterable(map(self.prepr.simple_preproc, sentences)))
        tokens = self.prepr._filter_rare_words(Counter(tokens), min_occurancies)

        unknown_keywords = []
        for w in [y for x in self.categories for y in x[1] if y not in tokens]:
            if w in self.navec:
                tokens.append(w)
            else:
                unknown_keywords.append(w)

        return (tokens, unknown_keywords)

    def create_navec_emb(self, inner_keys: List[str],
                                   random_seed: int, rand_uni_bound: float
                                   ) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
        np.random.seed(random_seed)

        inner_keys = ['<pad>', '<unk>'] + inner_keys
        input_dim = len(inner_keys)
        out_dim = self.navec.pq.dim

        vocab = {}
        matrix = np.zeros((input_dim, out_dim))
        unk_words = []

        for idx, word in enumerate(inner_keys):
            vocab[word] = idx
            if word in self.navec:
                matrix[idx] = self.navec[word]
            else:
                unk_words.append(word)
                matrix[idx] = np.random.uniform(-rand_uni_bound, rand_uni_bound, size=out_dim)
        matrix[0] = np.zeros_like(matrix[0])
        return matrix, vocab, unk_words

