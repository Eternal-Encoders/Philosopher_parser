import os

from dotenv import load_dotenv

from src.graphs import FileReader
from src.model_inf import OcrExec

load_dotenv()

ocr = OcrExec(
    os.environ['OCR_ENDPOINT'],
    os.environ['OPENROUTER_API_KEY'],
    os.environ['OCR_MODEL']
)
file_reader = FileReader(ocr.ocr, root_path='__output_TEST__')

def test_file_read():
    out_txt, out_imgs = file_reader.read_markdown(
        '__input_TEST__/test.docx',
        force_reload=True
    )

    assert len(out_imgs) == 2
    assert isinstance(out_txt, str)

def test_load():
    out_txt, out_imgs = file_reader.read_markdown(
        '__input_TEST__/test.docx',
        force_reload=False
    )
    out_txt, out_imgs = file_reader.read_markdown(
        '__input_TEST__/test.docx',
        force_reload=False
    )

    assert len(out_imgs) == 2
    assert isinstance(out_txt, str)

