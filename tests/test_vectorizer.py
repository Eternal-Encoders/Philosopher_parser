import os

import dotenv
import numpy as np

from src.model_inf import VectorizerExec

dotenv.load_dotenv()

vectorizer = VectorizerExec(
    os.environ['VECTORIZER_ENDPOINT'],
    os.environ['HUGGINGFACE_HUB_TOKEN'],
    os.environ['VECTORIZER_MODEL']
)
texts = [
    'То сё',
    'Пятое десятое'
]


def test_embed_single():
    out = vectorizer.encode_text(texts[0])
    
    assert isinstance(out, np.ndarray)
    assert len(out.shape) == 1


def test_embed():
    out = vectorizer.encode(texts)
    
    assert isinstance(out, np.ndarray)
    assert len(out.shape) == 2
    assert out.shape[0] == len(texts)
    assert out.shape[1] >= 200