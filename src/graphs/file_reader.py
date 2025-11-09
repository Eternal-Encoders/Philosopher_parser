import re
import base64
from markitdown import MarkItDown
from .models import ReaderImageOutput
from PIL import Image
from io import BytesIO
from typing import Tuple, List, Callable


class FileReader():
    @staticmethod
    def remove_images(
        text: str,
        imgs: list[ReaderImageOutput]
    ) -> str:
        for img in imgs:
            new_str = f'![{img.translation}](media/{img.image_name})'
            text = text.replace(img.image_uri, new_str)

        return text
    
    @staticmethod
    def __get_image(full_match: str) -> Image.Image:
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
        iters = re.finditer(r'!\[.*\]\(.+\)', text)
        visited = set[str]()
        images = []

        for i, img_str in enumerate(iters):
            img_str = img_str.group()

            if img_str not in visited:
                img  = FileReader.__get_image(img_str)
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
        self.md = MarkItDown() 
        self.ocr_fn = ocr_fn

    def read_markdown(self, file_path: str) -> Tuple[str, List[ReaderImageOutput]]:
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