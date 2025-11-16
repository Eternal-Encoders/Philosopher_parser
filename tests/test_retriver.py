import os
import numpy as np
from dotenv import load_dotenv
from src.graphs import Retriver
from src.model_inf import OcrExec, VectorizerExec, SummaryExec

load_dotenv()

ocr = OcrExec(
    os.environ['OPENROUTER_API_KEY'],
    os.environ['OCR_MODEL']
)
vectorizer = VectorizerExec(
    os.environ['HUGGINGFACE_HUB_TOKEN'],
    os.environ['VECTORIZER_MODEL']
)
summary_gen = SummaryExec(
    os.environ['OPENROUTER_API_KEY'],
    os.environ['SUMMARY_MODEL']
)

def test_embedding():
    parser = Retriver(
        f_path='__input_TEST__/test.docx',
        encode_q_fn=vectorizer.encode,
        encode_d_fn=vectorizer.encode,
        ocr_fn=ocr.ocr,
        gen_summary_fn=summary_gen.generate_summary_text,
        text_field='text',
        root_dir='__output_TEST__',
        force_reload=True
    )

    assert isinstance(parser.doc_emb, np.ndarray)
    assert isinstance(parser.node_ids, np.ndarray)
    assert len(parser.doc_emb) == len(parser.node_ids)

def test_emb_load():
    parser = Retriver(
        f_path='__input_TEST__/test.docx',
        encode_q_fn=vectorizer.encode,
        encode_d_fn=vectorizer.encode,
        ocr_fn=ocr.ocr,
        gen_summary_fn=summary_gen.generate_summary_text,
        text_field='text',
        root_dir='__output_TEST__',
        force_reload=False
    )
    parser = Retriver(
        f_path='__input_TEST__/test.docx',
        encode_q_fn=vectorizer.encode,
        encode_d_fn=vectorizer.encode,
        ocr_fn=ocr.ocr,
        gen_summary_fn=summary_gen.generate_summary_text,
        text_field='text',
        root_dir='__output_TEST__',
        force_reload=False
    )

    assert isinstance(parser.doc_emb, np.ndarray)
    assert isinstance(parser.node_ids, np.ndarray)
    assert len(parser.doc_emb) == len(parser.node_ids)