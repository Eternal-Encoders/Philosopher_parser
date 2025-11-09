import re
import gc
import base64
import torch
from markitdown import MarkItDown
from .models import ModelWrapper, ReaderImageOutput
from .utils import use_model_decorator
from PIL import Image
from io import BytesIO
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm.notebook import trange
from typing import Tuple, List

image_mask = r'!\[\]\(data:image/(png|jpeg);base64,.+\)'


def get_uri(full_match: str) -> str:
    """
    Извлекает URI изображения из полного совпадения строки.

    Args:
        full_match (str): Полная строка, содержащая URI изображения в формате base64.

    Returns:
        str: Извлеченный URI изображения.
    """
    uri_mask = r'(!\[\]\(data:image/(png|jpeg);base64,)|(\))'
    return re.sub(uri_mask, '', full_match)


def get_image(uri: str) -> Image.Image:
    """
    Декодирует URI изображения в формате base64 и возвращает объект PIL Image.

    Args:
        uri (str): URI изображения в формате base64.

    Returns:
        Image.Image: Объект PIL Image.
    """
    return Image.open(
        BytesIO(
            base64.b64decode(uri)
        )
    )


def get_msg(img: Image.Image):
    """
    Формирует сообщение для модели на основе изображения.

    Args:
        img (Image.Image): Объект PIL Image.

    Returns:
        list: Список сообщений в формате, ожидаемом моделью.
    """
    messages = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'image',
                    'image': img,
                },
                {
                    'type': 'text',
                    'text': 'Translate image to text, save core meaning, include all text from image in your responce'
                },
            ],
        }
    ]

    return messages


def retrive_images(text: str) -> List[ReaderImageOutput]:
    """
    Извлекает изображения из текста в формате base64 URI.

    Args:
        text (str): Входной текст, содержащий изображения в формате base64 URI.

    Returns:
        List[ReaderImageOutput]: Список объектов ReaderImageOutput, содержащих информацию об изображениях.
    """
    iters = re.finditer(image_mask, text)
    visited = set()
    images = []

    for i, img_str in enumerate(iters):
        img_str = img_str.group()

        if img_str not in visited:
            uri = get_uri(img_str)
            img  = get_image(uri)
            img_name = f'image_{i}.png'

            images.append(ReaderImageOutput(
                img_name,
                img,
                img_str,
            ))
            visited.add(img_str)

    return images


def remove_images(text: str, images: List[ReaderImageOutput], translations: List[str]) -> str:
    """
    Заменяет URI изображений в тексте на новые строки с переводами и именами файлов.

    Args:
        text (str): Входной текст, содержащий URI изображений.
        images (List[ReaderImageOutput]): Список объектов ReaderImageOutput, содержащих информацию об изображениях.
        translations (List[str]): Список переводов для каждого изображения.

    Returns:
        str: Текст с замененными URI изображений.
    """
    for img, text in zip(images, translations):
        new_str = f'![{text}](media/{img.image_name})'
        text = text.replace(img.image_uri, new_str)
    return text


class FileReader(ModelWrapper):
    """
    Класс для чтения файлов Markdown, извлечения изображений и получения их текстовых представлений с помощью модели Qwen2.5-VL-3B-Instruct.
    """
    def __init__(
        self,
        model_path='Qwen/Qwen2.5-VL-3B-Instruct'
    ) -> None:
        """
        Инициализирует FileReader.

        Args:
            model_path (str, optional): Путь к предварительно обученной модели. По умолчанию 'Qwen/Qwen2.5-VL-3B-Instruct'.
        """
        super().__init__(model_path)
        self.md = MarkItDown()
    
    def set_model(self):
        """
        Загружает предварительно обученную модель и процессор, а также компилирует модель для оптимизации.
        """
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype='auto',
            device_map='auto'
        )
        self.processor= AutoProcessor.from_pretrained(
            self.model_path,
            use_fast=True
        )
        self.model = torch.compile(
            model,
            mode='max-autotune',
            fullgraph=True
        )
    
    def dispatch_model(self):
        """
        Выгружает модель и процессор из памяти, очищает кэш CUDA.
        """
        del self.model, self.processor
        gc.collect()
        torch.cuda.empty_cache()
    
    def __get_input(self, messages):
        """
        Подготавливает входные данные для модели из списка сообщений.

        Args:
            messages (list): Список сообщений, каждое из которых содержит текст и/или изображение.

        Returns:
            torch.Tensor: Тензор входных данных, готовый для подачи в модель.
        """
        texts = [
            self.processor.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True
            )
            for msg in messages
        ]
        image_inputs, _ = process_vision_info(messages)  #type: ignore

        return self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
    
    @use_model_decorator
    def get_images_str(self, imgs: List[ReaderImageOutput], batch_size=4):
        """
        Получает текстовые представления изображений с помощью модели.

        Args:
            imgs (List[ReaderImageOutput]): Список объектов ReaderImageOutput, содержащих изображения.
            batch_size (int, optional): Размер батча для обработки изображений. По умолчанию 4.

        Returns:
            List[str]: Список текстовых представлений изображений.
        """
        res = []
        for i in trange(0, len(imgs), batch_size):
            batch = imgs[i:i+batch_size]
            msgs = [
                get_msg(img.image)
                for img in batch
            ]
            inputs = self.__get_input(msgs)

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(
                    inputs.input_ids,
                    generated_ids
                )
            ]
            res.extend((
                t.replace('\n', '. ')
                for t in self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
            ))

        return res    

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
        images = retrive_images(result.text_content)
        with open('./__output__/study_fies_no_uri.md', 'r', encoding='utf-8') as f:
            res = f.read()
        # translations = self.get_images_str(images, batch_size=3)
        # res = remove_images(result.text_content, images, translations)

        return res, images


# with open('./__output__/study_fies.md', 'w', encoding='utf-8') as f:
#     f.writelines(result.text_content)

# with open('./__output__/study_fies_no_uri.md', 'w', encoding='utf-8') as f:
#     f.writelines(txt)

# for file_name, img,_ in images:
#     img.save(f'./__output__/media/{file_name}')