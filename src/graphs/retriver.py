import os
import pickle
import numpy as np
import networkx as nx
from .file_reader import FileReader
from .graph_parser import GraphParser
from ..models import Parser, Parser, TypeReturn
from typing import Tuple, Generator, Callable, Any, List
from PIL import Image

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


class Retriver(Parser):
    @staticmethod
    def get_ids_text(graph: nx.Graph) -> Tuple[list[str], np.ndarray]:
        node_ids = []
        node_texts = []

        for k, v in dict(graph.nodes.data()).items():
            node_ids.append(k)
            node_texts.append(v['text'])

        node_ids= np.array(node_ids)

        return node_texts, node_ids

    def __init__(
        self,
        encode_q_fn: Callable[[list[str]], np.ndarray],
        encode_d_fn: Callable[[list[str]], np.ndarray],
        ocr_fn: Callable[[list[Image.Image]], list[str]],
        gen_summary_fn: Callable[[str], str]
    ) -> None:
        self.encode_q_fn = encode_q_fn
        self.encode_d_fn = encode_d_fn

        self.file_reader = FileReader(ocr_fn)
        self.graph_parser = GraphParser('__output__/binaries', gen_summary_fn)

        self.graph: nx.Graph | None = None
        self.doc_emb: np.ndarray | None = None
        self.node_ids: np.ndarray | None = None
    
    def load_graph(self, f_path: str):
        graph_path = os.path.join(f_path, 'graph.pkl')
        emb_path = os.path.join(f_path, 'docs.pkl')
        ids_path = os.path.join(f_path, 'ids.pkl')

        if not (os.path.exists(graph_path) and os.path.exists(emb_path) and os.path.exists(ids_path)):
            print(f"Бинарные файлы графа не найдены в {f_path}. Инициализация графа...")
            self.prepare_doc_file(f_path)
            return

        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)
        with open(emb_path, 'rb') as f:
            self.doc_emb = pickle.load(f)
        with open(ids_path, 'rb') as f:
            self.node_ids = pickle.load(f)

    def __get_linked_type(self, node: str, t: TypeReturn):
        assert self.graph is not None

        res = []
        visited = set()
        for linked_node in self.graph[node]:
            linked_data = self.graph.nodes[linked_node]
            if linked_data['node_type'] == t and linked_data['text'] not in visited:
                res.append(linked_data)
                visited.add(linked_data['text'])
        return res

    def __collate_context(self, node_ids: str):
        assert self.graph is not None

        node = self.graph.nodes[node_ids]
        res = []

        for tag, type_ret, statement in collaters:
            linked_data = self.__get_linked_type(node_ids, type_ret)

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

    def get_document(self, node_ids: str):
        assert  self.graph is not None

        return f'''
            <context>
            {self.__collate_context(node_ids)}
            </context>
            <text>
            {self.graph.nodes[node_ids]['text']}
            </text>
        '''.replace('\t', '')

    def vectorize_graph(self, graph: nx.Graph) -> Tuple[np.ndarray, np.ndarray]:
        node_ids = []
        node_texts = []

        for k, v in dict(graph.nodes.data()).items():
            node_ids.append(k)
            node_texts.append(v['text'])

        node_ids= np.array(node_ids)

        return (
            self.encode_d_fn(
                node_texts
            ),
            node_ids
        )
    
    def prepare_doc_file(
        self, 
        file_path: str, 
        emb: np.ndarray | None=None, 
        node_ids: np.ndarray | None=None
    ):
        assert (emb is not None) == (node_ids is not None)

        graph_dir = os.path.dirname(file_path)
        emb_path = os.path.join(graph_dir, 'docs.pkl')
        ids_path = os.path.join(graph_dir, 'ids.pkl')

        # Check if embeddings already exist on disk
        if os.path.exists(emb_path) and os.path.exists(ids_path):
            with open(emb_path, 'rb') as f:
                self.doc_emb = pickle.load(f)
            with open(ids_path, 'rb') as f:
                self.node_ids = pickle.load(f)
            print(f"Эмбеддинги и ID узлов загружены из {emb_path} и {ids_path}")
            
            # Load graph as well if embeddings are loaded
            graph_path = os.path.join(graph_dir, 'graph.pkl')
            if os.path.exists(graph_path):
                with open(graph_path, 'rb') as f:
                    self.graph = pickle.load(f)
                print(f"Граф загружен из {graph_path}")
            return # Exit if embeddings are loaded

        txt, imgs = self.file_reader.read_markdown('__output__/study_fies.md')
        graph = self.graph_parser.text2graph(txt, imgs)
        self.graph_parser.add_edges(graph)

        if emb is None or node_ids is None:
            node_txt, node_ids = Retriver.get_ids_text(graph)
            emb = self.encode_d_fn(node_txt)

        self.graph = graph
        self.doc_emb = emb
        self.node_ids = node_ids

        os.makedirs(graph_dir, exist_ok=True)
        with open(emb_path, 'wb') as f:
            pickle.dump(self.doc_emb, f)
        with open(ids_path, 'wb') as f:
            pickle.dump(self.node_ids, f)
        print(f"Эмбеддинги и ID узлов сохранены в {emb_path} и {ids_path}")
        
    def retrive_docs(
        self,
        query: str | List[str] | np.ndarray,
        file_path: str | None=None
    ):
        if not isinstance(query, np.ndarray):
            query = self.encode_q_fn(query if isinstance(query, list) else [query])
        if self.graph is None:
            assert file_path is not None, "file_path must be provided if graph is not loaded."
            self.prepare_doc_file(file_path)
    
        if self.graph is None:
            assert file_path is not None
            self.prepare_doc_file(file_path)
        
        assert self.graph is not None, "Graph is not loaded after prepare_doc_file."
        assert self.doc_emb is not None, "Document embeddings are not loaded after prepare_doc_file."
        assert self.node_ids is not None, "Node IDs are not loaded after prepare_doc_file."

        sims = self.doc_emb @ query
        top_k_ids = np.argsort(sims).reshape(-1)[::-1][:2]
        top_k = self.node_ids[top_k_ids]

        return [
            self.get_document(e).strip()
            for e in top_k
        ]
    
    def get_data_by_neighbour(
        self,
        query: TypeReturn,
        equation:  Callable[[Any], bool]
    ) -> Generator[str, None, None]:        
        assert self.graph is not None
        assert self.doc_emb is not None
        assert self.node_ids is not None

        return (
            self.graph.nodes[node_id]['text']
            for node_id in self.graph
            if self.graph.nodes[node_id]['node_type'] == query and any([
                equation(n)
                for n in self.__get_linked_type(node_id, TypeReturn.TEXT)
            ])
        )
