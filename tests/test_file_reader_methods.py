import base64
from io import BytesIO

import pytest
from PIL import Image

from src.graphs import FileReader


@pytest.fixture
def sample_image():
    img = Image.new('RGB', (100, 100), color='red')
    return img


@pytest.fixture
def sample_base64_image():
    img = Image.new('RGB', (50, 50), color='blue')
    buffered = BytesIO()
    img.save(buffered, format='PNG')
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'data:image/png;base64,{img_str}'


def test_get_image(sample_base64_image):
    markdown_uri = f'  ![]({sample_base64_image})  '
    print(markdown_uri)
    
    result = FileReader.get_image(markdown_uri)
    
    assert isinstance(result, Image.Image)
    assert result.size == (50, 50)
