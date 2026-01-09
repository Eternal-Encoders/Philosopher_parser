from dataclasses import dataclass
from enum import Enum

from PIL.Image import Image


class TypeReturn(Enum):
    HEADING='heading'
    TEXT='text'
    TABLE='table'
    LIST='list'
    IMAGE='image'
    FOOTNOTE='footnote'


@dataclass
class ReaderImageOutput:
    image_name: str
    image: Image
    image_uri: str
    translation: str | None = None


@dataclass
class GraphCollatorOutput:
    obj_type: TypeReturn
    text: str
    image: Image | None
    summary: str | None = None