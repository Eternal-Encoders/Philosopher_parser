import os
import dotenv
from PIL import Image
from src.model_inf import OcrExec

dotenv.load_dotenv()

ocr = OcrExec(
    os.environ['OPENROUTER_API_KEY'],
    os.environ['OCR_MODEL']
)
images = [
    Image.open('./__output__/media/image_0.png'),
    Image.open('./__output__/media/image_1.png')
]


def test_ocr_single():
    out = ocr.ocr_image(images[0])
    
    assert isinstance(out, str)
    assert len(out) != 0

    print(out)


def test_ocr():
    out = ocr.ocr(images)  #type: ignore
    
    assert isinstance(out, list)
    assert all((
        isinstance(o, str)
        for o in out
    ))
    assert all((
        len(o) != 0
        for o in out
    ))
    assert len(images) == len(out)

    print(out)