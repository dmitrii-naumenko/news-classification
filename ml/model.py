import pandas as pd
import numpy as np
import random
import os
from ml.duplicates_processor import DuplicatesProcessor
from ml.classify_processor import ClassifyProcessor

class Model:
    def __init__(self):
        self.duplicate_model = DuplicatesProcessor()
        self.classify_model = ClassifyProcessor()

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Classify texts in data and clean for duplicates"""
        corpus = self._preprocess(data)
        corpus = self._del_duplicates(corpus)
        result = self._classify(corpus)
        return result

    def _classify(self, corpus: np.ndarray) -> np.ndarray:
        """Classify texts into categories."""
        result = self.classify_model.classify(corpus)
        return result

    def _del_duplicates(self, corpus: np.ndarray) -> np.ndarray:
        """Delete duplicated texts from corpus."""
        result = self.duplicate_model.drop_duplicated(corpus)
        return result

    def _preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """Preprocess input data"""
        ds = df.copy()
        ds.text = ds.text.astype(str)
        return np.unique(ds.text.to_numpy())
