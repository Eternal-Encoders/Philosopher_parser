import numpy as np
from enum import Enum
from typing import List, Callable, Any, Generator
from pydantic import BaseModel


class TypeReturn(Enum):
    HEADING='heading'
    TEXT='text'
    TABLE='table'
    LIST='list'
    IMAGE='image'
    FOOTNOTE='footnote'


class RAGModel(BaseModel):
    query: str = 'Какие существуют направления в философии?'


class Parser:
    def retrive_docs(
        self,
        query: str | List[str] | np.ndarray,
        file_path: str | None=None
    ) -> List[str]:
        raise NotImplementedError('Parser must implement this method one way or another')
    
    def get_data_by_neighbour(
        self,
        query: TypeReturn,
        equation:  Callable[[Any], bool]
    ) -> Generator[str, None, None]:
        raise NotImplementedError('Parser must implement this method one way or another')