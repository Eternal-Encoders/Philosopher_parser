from pydantic import BaseModel


class RAGModel(BaseModel):
    query: str = 'Какие существуют направления в философии?'