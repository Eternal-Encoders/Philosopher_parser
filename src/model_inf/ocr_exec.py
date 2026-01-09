import base64
import io
import json
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from PIL import Image
from tqdm.auto import tqdm


class OcrExec:
    @staticmethod
    def pil2base64(img: Image.Image):
        buffered = io.BytesIO()
        img.save(buffered, format='PNG')
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return f'data:image/png;base64,{img_str}'
    
    @staticmethod
    def req2jsonl(reqs: list[dict[str, Any]]):
        reqs_str = [json.dumps(req) for req in reqs]
        buffer = io.StringIO()
        buffer.writelines(reqs_str)

        return buffer

    def __init__(
        self,
        endpoint: str,
        openrouter_key: str,
        name_or_path: str,
        site_url: str | None=None,
        site_title: str | None=None
    ) -> None:
        self.client = OpenAI(
            base_url=endpoint,
            api_key=openrouter_key,
        )
        self.name_or_path = name_or_path
        self.extra_headers = {}

        if site_url is not None:
            self.extra_headers.update({
                'HTTP-Referer': site_url
            })
        if site_title is not None:
            self.extra_headers.update({
                'X-Title': site_title
            })
    
    def __get_msg(self, img: Image.Image) -> ChatCompletionMessageParam:
        return {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': 'Translate image to text, save core meaning, \
                        include all text from image in your responce'
                },
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': OcrExec.pil2base64(img)
                    }
                }
            ]
        }

    def ocr_image(self, img: Image.Image):
        completion = self.client.chat.completions.create(
            extra_headers=self.extra_headers,
            extra_body={'reasoning': {'enabled': False}},
            model=self.name_or_path,
            messages=[self.__get_msg(img)],
            max_completion_tokens=1024,
            temperature=0.01
        )

        return completion.choices[0].message.content

    def ocr(self, imgs: list[Image.Image]):
        res = []
        for i in tqdm(imgs):
            res.append(self.ocr_image(i))
        return res