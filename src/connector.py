import numpy as np
from .models import Parser
from typing import List


class Connector:
    def __init__(self, parser: Parser) -> None:
        self.parser = parser
    
    def get_docs(
        self,
        query: str | List[str] | np.ndarray,
        file_path: str | None=None
    ):
        res = self.parser.retrive_docs(query, file_path)

        return res