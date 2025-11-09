from PIL.Image import Image
from dataclasses import dataclass
from ..models import TypeReturn


class ModelWrapper:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = None

    def set_model(self):
        raise NotImplementedError()

    def dispatch_model(self):
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
    summary: str | None = None