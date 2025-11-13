import re
import base64
from markitdown import MarkItDown
from .models import ReaderImageOutput
from PIL import Image
from io import BytesIO
from typing import Tuple, List, Callable


class FileReader():
    """
    Класс для чтения файлов Markdown, извлечения изображений и получения их текстовых представлений с помощью модели Qwen2.5-VL-3B-Instruct.
    """

    @staticmethod
    def remove_images(
        text: str,
        imgs: list[ReaderImageOutput]
    ) -> str:
        """
        Заменяет URI изображений в тексте на новые строки с переводами и именами файлов.

        Args:
            text (str): Входной текст, содержащий URI изображений.
            images (List[ReaderImageOutput]): Список объектов ReaderImageOutput, содержащих информацию об изображениях.

        Returns:
            str: Текст с замененными URI изображений.
        """
        for img in imgs:
            new_str = f'![{img.translation}](media/{img.image_name})'
            text = text.replace(img.image_uri, new_str)

        return text
    
    @staticmethod
    def get_image(full_match: str) -> Image.Image:
        """
        Декодирует URI изображения в формате base64 и возвращает объект PIL Image.

        Args:
            full_match (str): Полная строка, содержащая URI изображения в формате base64.

        Returns:
            Image.Image: Объект PIL Image.
        """
        full_match = full_match.strip()
        full_match = re.sub(r'^!\[.+\]\(', '', full_match).rstrip(')')
        uri = re.sub(r'^data:image/.+;base64,', '', full_match)

        return Image.open(
            BytesIO(
                base64.b64decode(uri)
            )
        )
    
    @staticmethod
    def retrive_images(text: str) -> List[ReaderImageOutput]:
        """
        Извлекает изображения из текста в формате base64 URI.

        Args:
            text (str): Входной текст, содержащий изображения в формате base64 URI.

        Returns:
            List[ReaderImageOutput]: Список объектов ReaderImageOutput, содержащих информацию об изображениях.
        """
        iters = re.finditer(r'!\[.*\]\(.+\)', text)
        visited = set[str]()
        images = []

        for i, img_str in enumerate(iters):
            img_str = img_str.group()

            if img_str not in visited:
                img  = FileReader.get_image(img_str)
                img_name = f'image_{i}.png'

                images.append(ReaderImageOutput(
                    image_name=img_name,
                    image=img,
                    image_uri=img_str,
                ))
                visited.add(img_str)

        return images

    def __init__(
        self,
        ocr_fn: Callable[[List[Image.Image]], List[str]]
    ) -> None:
        """
        Инициализирует FileReader.

        Args:
            ocr_fn (Callable[[List[Image.Image]], List[str]]): Функция для распознования изображения.
        """
        self.md = MarkItDown() 
        self.ocr_fn = ocr_fn

    def read_markdown(self, file_path: str) -> Tuple[str, List[ReaderImageOutput]]:
        """
        Читает файл Markdown, извлекает изображения, получает их текстовые представления и заменяет URI изображений в тексте.

        Args:
            file_path (str): Путь к файлу Markdown.

        Returns:
            Tuple[str, List[ReaderImageOutput]]: Кортеж, содержащий текст Markdown с замененными URI изображений и список объектов ReaderImageOutput.
        """
        result = self.md.convert(
            file_path,
            keep_data_uris=True
        )
        images = FileReader.retrive_images(result.text_content)

        for i, tr in enumerate(self.ocr_fn([
            img.image
            for img in images
        ])):
            images[i].translation = tr
        
        return FileReader.remove_images(
            result.text_content,
            images,
        ), images


# with open('./__output__/study_fies.md', 'w', encoding='utf-8') as f:
#     f.writelines(result.text_content)

# with open('./__output__/study_fies_no_uri.md', 'w', encoding='utf-8') as f:
#     f.writelines(txt)

# for file_name, img,_ in images:
#     img.save(f'./__output__/media/{file_name}')