from PIL.Image import Image
from dataclasses import dataclass
from ..models import TypeReturn


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