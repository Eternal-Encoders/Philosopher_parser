import numpy as np
from openai import OpenAI

prompts = {
    'query': 'task: search result | query: ',
    'document': 'title: none | text: ',
    'BitextMining': 'task: search result | query: ',
    'Clustering': 'task: clustering | query: ',
    'Classification': 'task: classification | query: ',
    'InstructionRetrieval': 'task: code retrieval | query: ',
    'MultilabelClassification': 'task: classification | query: ',
    'PairClassification': 'task: sentence similarity | query: ',
    'Reranking': 'task: search result | query: ',
    'Retrieval': 'task: search result | query: ',
    'Retrieval-query': 'task: search result | query: ',
    'Retrieval-document': 'title: none | text: ',
    'STS': 'task: sentence similarity | query: ',
    'Summarization': 'task: summarization | query: '
}


class VectorizerExec():
    def __init__(
        self,
        hf_key: str,
        name_or_path: str,
    ) -> None:
        self.client = OpenAI(
            base_url='https://roaoch-vectorizer.hf.space/v1',
            api_key=hf_key,
        )
        self.name_or_path = name_or_path

    def encode_text(self, text: str, prompt_name='query') -> np.ndarray:
        out = self.client.embeddings.create(
            input=prompts[prompt_name] + text,
            model=self.name_or_path
        )

        return np.array(out.data[0].embedding)

    def encode(self, texts: list[str], prompt_name='document'):
        out = self.client.embeddings.create(
            input=[
                prompts[prompt_name] + t
                for t in texts
            ],
            model=self.name_or_path
        )

        return np.array([
            o.embedding
            for o in out.data
        ])