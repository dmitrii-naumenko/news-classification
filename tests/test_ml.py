import pandas as pd
import os

from ml.model import Model
from ml.duplicates_processor import DuplicatesProcessor


def test_init():
    model = Model()


def test_process():
    model = Model()
    path = os.path.join(os.path.dirname(__file__), 'test_1000.xlsx')
    df = pd.read_excel(path)
    result = model.process(df)
    result.to_excel("clean_data.xlsx")
