from pydantic import BaseModel


class RAGModel(BaseModel):
    query: str = 'Какие существуют направления в философии?'
    top_k: int = 2
    max_length: int = 2000