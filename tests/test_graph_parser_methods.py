from unittest.mock import MagicMock

import pytest
from PIL import Image

from src.graphs import GraphParser
from src.graphs.models import ReaderImageOutput, TypeReturn


@pytest.fixture
def sample_images():
    return [
        ReaderImageOutput(
            image_name='media/image_0.png',
            image=Image.new('RGB', (10, 10)),
            image_uri='![alt](uri)',
            translation='translated text'
        )
    ]


def test_get_type_return_heading():
    assert GraphParser.get_type_return('# Heading') == TypeReturn.HEADING
    assert GraphParser.get_type_return('## Subheading') == TypeReturn.HEADING
    assert GraphParser.get_type_return('### Deep heading') \
        == TypeReturn.HEADING


def test_get_type_return_table():
    assert GraphParser.get_type_return('| col1 | col2 |') == TypeReturn.TABLE
    assert GraphParser.get_type_return('| a | b | c |') == TypeReturn.TABLE


def test_get_type_return_list():
    assert GraphParser.get_type_return('1. First item') == TypeReturn.LIST
    assert GraphParser.get_type_return('* Bullet point') == TypeReturn.LIST


def test_get_type_return_image():
    assert GraphParser.get_type_return('![alt](path.png)') == TypeReturn.IMAGE


def test_get_type_return_footnote():
    assert GraphParser.get_type_return('[[1]](#footnote-1)') \
        == TypeReturn.FOOTNOTE


def test_get_type_return_text():
    assert GraphParser.get_type_return('Plain text') == TypeReturn.TEXT
    assert GraphParser.get_type_return('No special format') == TypeReturn.TEXT
    assert GraphParser.get_type_return('Text *with* asterisks') \
        == TypeReturn.TEXT


def test_to_imgs_base(sample_images: list[ReaderImageOutput]):
    result = GraphParser.to_imgs_base(sample_images)
    
    assert isinstance(result, dict)
    assert sample_images[0].image_name in result
    assert result['media/image_0.png'] == sample_images[0]


def test_get_level(sample_images):
    level, text = GraphParser.get_level('# Heading')
    assert level == 1
    assert text == 'Heading'
    
    level, text = GraphParser.get_level('## Subheading')
    assert level == 2
    assert text == 'Subheading'
    
    level, text = GraphParser.get_level('### Deep')
    assert level == 3
    assert text == 'Deep'

    level, img = GraphParser.get_level(sample_images[0].image)
    assert level == -1
    assert isinstance(img, Image.Image)

    level, text = GraphParser.get_level('Plain text')
    assert level == -1
    assert text == 'Plain text'


def test_collate_f_simple():
    mock_summary = MagicMock(return_value='summary')
    parser = GraphParser(
        '__output_TEST__/binaries',
        mock_summary,
        generate_summary=False
    )
    
    text_lines = ['# Heading', 'Plain text', 'More text']
    imgs = {}
    
    results = list(parser.collate_f(text_lines, imgs))
    
    assert len(results) == 3
    assert results[0].obj_type == TypeReturn.HEADING
    assert results[1].obj_type == TypeReturn.TEXT
    assert results[2].obj_type == TypeReturn.TEXT


def test_collate_f_with_table():
    mock_summary = MagicMock(return_value='summary')
    parser = GraphParser(
        '__output_TEST__/binaries',
        mock_summary,
        generate_summary=False
    )
    
    text_lines = [
        '| col1 | col2 |',
        '| val1 | val2 |',
        'Plain text'
    ]
    imgs = {}
    
    results = list(parser.collate_f(text_lines, imgs))
    
    assert len(results) >= 1
    assert len([r for r in results if r.obj_type == TypeReturn.TABLE]) == 1


def test_collate_f_with_list():
    mock_summary = MagicMock(return_value='summary')
    parser = GraphParser(
        '__output_TEST__/binaries',
        mock_summary,
        generate_summary=False
    )

    text_lines = [
        '1. First item',
        '2. Second item',
        '3. Third item',
        'Plain text'
    ]
    imgs = {}

    results = list(parser.collate_f(text_lines, imgs))

    list_results = [r for r in results if r.obj_type == TypeReturn.LIST]

    assert len(list_results) == 1


def test_collate_f_with_image(sample_images: list[ReaderImageOutput]):
    mock_summary = MagicMock(return_value='summary')
    parser = GraphParser(
        '__output_TEST__/binaries',
        mock_summary,
        generate_summary=False
    )
    
    text_lines = ['![alt](media/image_0.png)', 'Plain text']
    imgs = {'media/image_0.png': sample_images[0]}
    
    results = list(parser.collate_f(text_lines, imgs))
    
    image_results = [r for r in results if r.obj_type == TypeReturn.IMAGE]

    assert len(image_results) == 1
    assert image_results[0].image is not None


def test_collate_f_skips_empty():
    mock_summary = MagicMock(return_value='summary')
    parser = GraphParser(
        '__output_TEST__/binaries',
        mock_summary,
        generate_summary=False
    )
    
    text_lines = ['# Heading', '', '   ', 'Plain text']
    imgs = {}

    results = list(parser.collate_f(text_lines, imgs))

    assert len(results) == 2