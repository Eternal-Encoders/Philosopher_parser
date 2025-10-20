import gc
import os
import torch
import pickle
import numpy as np
import networkx as nx
from .file_reader import FileReader
from .graph_parser import GraphParser
from .models  import ModelWrapper
from ..models import TypeReturn
from .utils import use_model_decorator
from ..models import Parser
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

collaters = [
    (
        'images',
        TypeReturn.IMAGE,
        None
    ),
    (
        'parents',
        TypeReturn.HEADING,
        lambda root_node, to_node: root_node['level'] > to_node['level']
    ),
    (
        'childs',
        TypeReturn.HEADING,
        lambda root_node, to_node: root_node['level'] < to_node['level']
    ),
    (
        'lists',
        TypeReturn.LIST,
        None
    ),
    (
        'tables',
        TypeReturn.TABLE,
        None
    ),
    (
        'links',
        TypeReturn.FOOTNOTE,
        None
    ),
    (
        'view',
        TypeReturn.TEXT,
        None
    )
]


def __get_linked_type(graph: nx.Graph, node: str, t: TypeReturn):
    res = []
    visited = set()
    for linked_node in graph[node]:
        linked_data = graph.nodes[linked_node]
        if linked_data['node_type'] == t and linked_data['text'] not in visited:
            res.append(linked_data)
            visited.add(linked_data['text'])
    return res


def __collate_context(graph, node_ids: str):
    node = graph.nodes[node_ids]
    res = []

    for tag, type_ret, statement in collaters:
        linked_data = __get_linked_type(graph, node_ids, type_ret)

        if statement is not None:
            linked_data = [
                l
                for l in linked_data
                if statement(node, l)
            ]

        linked_data = '\n'.join((
            l['text']
            for l in linked_data
        ))
        res.append(f'<{tag}>\n{linked_data}\n</{tag}>')
    return '\n'.join(res)


def get_document(graph, node_ids: str):
        return f'''
        <context>
        {__collate_context(graph, node_ids)}
        </context>
        <text>
        {graph.nodes[node_ids]['text']}
        </text>
    '''.replace('\t', '')


class GraphRetriver(ModelWrapper, Parser):
    def __init__(
        self,
        model_path='google/embeddinggemma-300m'
    ) -> None:
        super().__init__(model_path)

        self.file_reader = FileReader()
        self.graph_parser = GraphParser()

        self.graph: nx.Graph | None = None
        self.doc_emb: np.ndarray | None = None
        self.node_ids: np.ndarray | None = None
    
    def set_model(self):
        self.model = SentenceTransformer(
            self.model_path,
            model_kwargs={
                'dtype': torch.bfloat16,
                'device_map': 'auto'
            }
        )
        self.model = torch.compile(
            self.model,
            mode='max-autotune',
            fullgraph=True
        )
    
    def dispatch_model(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    def load_graph(self, f_path: str):
        graph_path = f'{f_path}/graph.pkl'
        emb_path = f'{f_path}/docs.pkl'
        ids_path = f'{f_path}/ids.pkl'

        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)
        if os.path.exists(emb_path) and os.path.exists(ids_path):
            with open(emb_path, 'rb') as f:
                self.doc_emb = pickle.load(f)
            with open(ids_path, 'rb') as f:
                self.node_ids = pickle.load(f)

    @use_model_decorator
    def vectorize_graph(self, graph: nx.Graph) -> Tuple[np.ndarray, np.ndarray]:
        node_ids = []
        node_texts = []

        for k, v in dict(graph.nodes.data()).items():
            node_ids.append(k)
            node_texts.append(v['text'])

        node_ids= np.array(node_ids)
        node_texts= np.array(node_texts)

        return (
            self.model.encode(
                node_texts,
                prompt_name='document',
                batch_size=16,
                show_progress_bar=True,
                normalize_embeddings=True
            ),
            node_ids
        )
    
    @use_model_decorator
    def vectorize_search(self, query: str | List[str]) -> np.ndarray:
        search_emb = self.model.encode(
            query,
            prompt_name='query',
            normalize_embeddings=True
        )

        return search_emb

    def prepare_doc_file(
        self, 
        file_path: str, 
        emb: np.ndarray | None=None, 
        node_ids: np.ndarray | None=None
    ):
        assert (emb is not None) == (node_ids is not None)

        txt, imgs = self.file_reader.read_markdown(file_path)
        graph = self.graph_parser.text2graph(txt, imgs)
        self.graph_parser.add_edges(graph)

        if emb is None or node_ids is None:
            emb, node_ids = self.vectorize_graph(graph)

        self.graph = graph
        self.doc_emb = emb
        self.node_ids = node_ids
        
    def retrive_docs(
        self,
        query: str | List[str] | np.ndarray,
        file_path: str | None=None
    ):
        if not isinstance(query, np.ndarray):
            query = self.vectorize_search(query)
        if self.graph is None:
            assert file_path is not None
            self.prepare_doc_file(file_path)
        
        assert self.graph is not None
        assert self.doc_emb is not None
        assert self.node_ids is not None

        sims = self.doc_emb @ query
        top_k_ids = np.argsort(sims).reshape(-1)[::-1][:5]
        top_k = self.node_ids[top_k_ids]

        return [
            get_document(self.graph, e).strip()
            for e in top_k
        ]