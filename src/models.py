from enum import Enum
from PIL.Image import Image
from dataclasses import dataclass


class TypeReturn(Enum):
    HEADING='heading'
    TEXT='text'
    TABLE='table'
    LIST='list'
    IMAGE='image'
    FOOTNOTE='footnote'


class ModelWrapper:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = None

    def __set_model(self):
        raise NotImplementedError()

    def __dispatch_model(self):
        raise NotImplementedError()


@dataclass
class ReaderImageOutput:
    image_name: str
    image: Image
    image_uri: str


@dataclass
class GraphCollatorOutput:
    obj_type: TypeReturn
    text: str
    image: Image | None