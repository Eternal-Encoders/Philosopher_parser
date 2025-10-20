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


def __get_uri(full_match: str) -> str:
    uri_mask = r'(!\[\]\(data:image/(png|jpeg);base64,)|(\))'
    return re.sub(uri_mask, '', full_match)


def __get_image(uri: str) -> Image.Image:
    return Image.open(
        BytesIO(
            base64.b64decode(uri)
        )
    )


def __get_msg(img: Image.Image):
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
    iters = re.finditer(image_mask, text)
    visited = set()
    images = []

    for i, img_str in enumerate(iters):
        img_str = img_str.group()

        if img_str not in visited:
            uri = __get_uri(img_str)
            img  = __get_image(uri)
            img_name = f'image_{i}.png'

            images.append(ReaderImageOutput(
                img_name,
                img,
                img_str,
            ))
            visited.add(img_str)

    return images


def remove_images(text: str, images: List[ReaderImageOutput], translations: List[str]) -> str:
    for img, text in zip(images, translations):
        new_str = f'![{text}](media/{img.image_name})'
        text = text.replace(img.image_uri, new_str)
    return text


class FileReader(ModelWrapper):
    def __init__(
        self,
        model_path='Qwen/Qwen2.5-VL-3B-Instruct'
    ) -> None:
        super().__init__(model_path)
        self.md = MarkItDown()
    
    def set_model(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype='auto',
            device_map='auto'
        )
        self.processor= AutoProcessor.from_pretrained(
            self.model_path,
            use_fast=True
        )
        self.model = torch.compile(
            self.model,
            mode='max-autotune',
            fullgraph=True
        )
    
    def dispatch_model(self):
        del self.model, self.processor
        gc.collect()
        torch.cuda.empty_cache()
    
    def __get_input(self, messages):
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
        res = []
        for i in trange(0, len(imgs), batch_size):
            batch = imgs[i:i+batch_size]
            msgs = [
                __get_msg(img.image)
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
        result = self.md.convert(
            file_path,
            keep_data_uris=True
        )
        images = retrive_images(result.text_content)
        translations = self.get_images_str(images, batch_size=3)
        
        return remove_images(result.text_content, images, translations), images


# with open('./__output__/study_fies.md', 'w', encoding='utf-8') as f:
#     f.writelines(result.text_content)

# with open('./__output__/study_fies_no_uri.md', 'w', encoding='utf-8') as f:
#     f.writelines(txt)

# for file_name, img,_ in images:
#     img.save(f'./__output__/media/{file_name}')