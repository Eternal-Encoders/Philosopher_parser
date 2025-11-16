import os
import re
import base64
import pickle
from markitdown import MarkItDown
from .models import ReaderImageOutput
from PIL import Image
from io import BytesIO
from typing import Callable


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
            new_str = f'![{img.translation}]({img.image_name})'
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
        full_match = full_match.lstrip('![](').rstrip(')')
        uri = re.sub(r'^data:image/.+;base64,', '', full_match)

        return Image.open(
            BytesIO(
                base64.b64decode(uri)
            )
        )
    
    @staticmethod
    def retrive_images(text: str) -> list[ReaderImageOutput]:
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
                img_name = f'media/image_{i}.png'

                images.append(ReaderImageOutput(
                    image_name=img_name,
                    image=img,
                    image_uri=img_str
                ))
                visited.add(img_str)

        return images

    def __init__(
        self,
        ocr_fn: Callable[[list[Image.Image]], list[str]],
        root_path='__output__'
    ) -> None:
        """
        Инициализирует FileReader.

        Args:
            ocr_fn (Callable[[List[Image.Image]], List[str]]): Функция для распознования изображения.
        """
        self.md = MarkItDown() 
        self.ocr_fn = ocr_fn

        self.root_path = root_path
        os.makedirs(self.root_path, exist_ok=True)
        self.md_path = os.path.join(root_path, 'study_fies.md')
        self.img_path = os.path.join(root_path, 'binaries', 'images.pkl')

    def read_markdown(self, file_path: str, force_reload=False):
        """
        Читает файл Markdown, извлекает изображения, получает их текстовые представления и заменяет URI изображений в тексте.
        Если файлы существуют, загружает их. В противном случае создает новые объекты и сохраняет их.

        Args:
            file_path (str): Путь к файлу Markdown.

        Returns:
            Tuple[str, List[ReaderImageOutput]]: Кортеж, содержащий текст Markdown с замененными URI изображений и список объектов ReaderImageOutput.
        """
        if os.path.exists(self.md_path) and os.path.exists(self.img_path) and not force_reload:
            return self.load_doc()

        self.clear_saved_data()

        result = self.md.convert(
            file_path,
            keep_data_uris=True
        )
        images = FileReader.retrive_images(result.text_content)

        for i, tr in enumerate(self.ocr_fn([
            img.image
            for img in images
        ])):
            images[i].translation = tr.replace('\n', ' ')
        
        txt = FileReader.remove_images(
            result.text_content,
            images,
        )

        self.save_doc(txt, images)

        return txt, images
    
    def load_doc(self) -> tuple[str, list[ReaderImageOutput]]:
        """
        Загружает сохраненный файл документа и изображений
        """
        with open(self.md_path, 'r', encoding='utf-8') as f:
            txt = '\n'.join(f.readlines())
        with open(self.img_path, 'rb') as f:
            images = pickle.load(f)
        
        return txt, images
        
    def save_doc(self, txt: str, images: list[ReaderImageOutput]):
        """
        Сохраняет в файл документв в формате Markdown и изображения
        """
        with open(self.md_path, 'w', encoding='utf-8') as f:
            f.writelines(txt)
        os.makedirs(os.path.dirname(self.img_path), exist_ok=True)
        with open(self.img_path, 'wb') as f:
            pickle.dump(images, f)

        for img in images:
            pp = os.path.join(self.root_path, img.image_name)
            os.makedirs(os.path.dirname(pp), exist_ok=True)
            img.image.save(pp)
    
    def clear_saved_data(self):
        """
        Удаляет сохраненные файлы
        """
        if os.path.exists(self.md_path):
            os.remove(self.md_path)
            print(f'Файл {self.md_path} Удален')
        if os.path.exists(self.img_path):
            os.remove(self.img_path)
            print(f'Файл {self.img_path} Удален')
        
        media_files = os.path.join(self.root_path, 'media')
        if os.path.exists(media_files):
            for f in os.listdir(media_files):
                os.remove(os.path.join(media_files, f))
