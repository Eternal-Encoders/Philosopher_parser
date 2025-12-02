import os
import re
import pickle
import numpy as np
import networkx as nx
from .models import TypeReturn
from .file_reader import FileReader
from .graph_parser import GraphParser
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


class Retriver():
    @staticmethod
    def get_ids_text(graph: nx.Graph, text_field='text') -> Tuple[list[str], np.ndarray]:
        node_ids = []
        node_texts = []

        for k, v in dict(graph.nodes.data()).items():
            node_ids.append(k)
            node_texts.append(v[text_field])

        node_ids= np.array(node_ids)

        return node_texts, node_ids

    def __init__(
        self,
        f_path: str,
        encode_q_fn: Callable[[list[str]], np.ndarray],
        encode_d_fn: Callable[[list[str]], np.ndarray],
        ocr_fn: Callable[[list[Image.Image]], list[str]],
        gen_summary_fn: Callable[[str], str],
        text_field='text',
        root_dir='__output__',
        force_reload=False
    ) -> None:
        self.encode_q_fn = encode_q_fn
        self.encode_d_fn = encode_d_fn

        self.file_reader = FileReader(ocr_fn, root_path=root_dir)
        self.graph_parser = GraphParser(
            f'{root_dir}/binaries',
            gen_summary_fn
        )

        txt, imgs = self.file_reader.read_markdown(
            f_path,
            force_reload=force_reload
        )
        self.graph = self.graph_parser.text2graph(
            txt,
            imgs,
            force_reload=force_reload
        )
        self.graph_parser.add_edges(self.graph, window_size=3)

        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

        self.doc_emb_path = os.path.join(root_dir, 'binaries', 'docs.pkl')
        self.nodes_ids_path = os.path.join(root_dir, 'binaries', 'ids.pkl')

        self.doc_emb, self.node_ids = self.prepare_data(
            force_reload=force_reload,
            text_field=text_field
        )
    
    def prepare_data(
        self,
        text_field: str,
        force_reload=False,
        emb: np.ndarray | None=None, 
        node_ids: np.ndarray | None=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Векторизует входной граф
        Если файл эмбеддинга существует, загружает его. В противном случае создает новый эмбеддинг и сохраняет его.
        """
        if os.path.exists(self.doc_emb_path) and os.path.exists(self.nodes_ids_path) and not force_reload:
            return self.load_data()
        
        self.clear_data()

        if emb is None or node_ids is None:
            node_txt, node_ids = Retriver.get_ids_text(self.graph, text_field=text_field)
            emb = self.encode_d_fn(node_txt)
        
        self.save_data(emb, node_ids)

        return emb, node_ids

    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Загружает данные эмбеддинга из файла
        """
        with open(self.doc_emb_path, 'rb') as f:
            doc_emb = pickle.load(f)
        with open(self.nodes_ids_path, 'rb') as f:
            node_ids = pickle.load(f)
        
        return doc_emb, node_ids
    
    def save_data(self, doc_emb: np.ndarray, node_ids: np.ndarray):
        """
        Сохраняет данные эмбеддинга в файл
        """
        with open(self.doc_emb_path, 'wb') as f:
            pickle.dump(doc_emb, f)
        print(f'Эмбединг сохранен в {self.doc_emb_path}')
        with open(self.nodes_ids_path, 'wb') as f:
            pickle.dump(node_ids, f)
        print(f'Id графа сохранен в {self.nodes_ids_path}')

    def clear_data(self):
        """
        Удаляет сохранённые файлы эмбеддингов
        """
        if os.path.exists(self.doc_emb_path):
            os.remove(self.doc_emb_path)
            print(f'Файл эмбеддингов удален: {self.doc_emb_path}')
        if os.path.exists(self.nodes_ids_path):
            os.remove(self.nodes_ids_path)
            print(f'Файл ID узлов удален: {self.nodes_ids_path}')

    def __get_linked_type(self, node: str, t: TypeReturn):
        res = []
        visited = set()
        for linked_node in self.graph[node]:
            linked_data = self.graph.nodes[linked_node]
            if linked_data['node_type'] == t and linked_data['text'] not in visited:
                res.append(linked_data)
                visited.add(linked_data['text'])
        return res

    def __collate_context(self, node_ids: str):
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
        return f'''
            <context>
            {self.__collate_context(node_ids)}
            </context>
            <text>
            {self.graph.nodes[node_ids]['text']}
            </text>
        '''.replace('\t', '')

    def get_similar(self, query: np.ndarray):
        sims = np.einsum('ij,j->i', self.doc_emb, query)
        top_k_ids = np.argsort(sims).reshape(-1)[::-1][:2]
        top_k = self.node_ids[top_k_ids]

        return top_k

    def retrive_docs(
        self,
        query: str | List[str] | np.ndarray
    ):
        if not isinstance(query, np.ndarray):
            # Пока временное решение, пока передаем по одному запросу
            query = self.encode_q_fn(query if isinstance(query, list) else [query]).reshape(-1)

        return [
            self.get_document(e).strip()
            for e in self.get_similar(query)
        ]
    
    def get_questions(self):
        lists = {
            re.sub(r'^(\d+.)+ ', '', e)
            for node_id in self.graph
            if self.graph.nodes[node_id]['node_type'] == TypeReturn.LIST and any([
                ('вопросы для обсуждения' in n['text'].lower() or \
                    'контрольные вопросы' in n['text'].lower() or \
                    'задание' in n['text'].lower()) and self.graph.nodes[node_id]['position'] - n['position'] <= 1
                for n in self.__get_linked_type(node_id, TypeReturn.TEXT)
            ])
            for e in self.graph.nodes[node_id]['text'].split('\n')
        }

        return lists
