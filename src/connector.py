import re
import numpy as np
from .models import Parser, TypeReturn
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
    
    def get_questions(
        self
    ):
        lists = {
            re.sub(r'^(\d+.)+ ', '', e)
            for es in self.parser.get_data_by_neighbour(
                TypeReturn.LIST,
                lambda node: 'вопрос' in node['text'].lower() and '**' in node['text']
            )
            for e in es.split('\n')
        }

        return lists