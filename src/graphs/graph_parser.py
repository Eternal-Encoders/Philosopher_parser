import os
import pickle
import re
from collections.abc import Callable, Iterable
from copy import deepcopy
from itertools import combinations
from uuid import uuid4

import networkx as nx
from PIL import Image
from tqdm import tqdm

from .models import GraphCollatorOutput, ReaderImageOutput, TypeReturn

re2enum = {
    r'^#+ .+': TypeReturn.HEADING,
    r'(\| .+ \| ?)+': TypeReturn.TABLE,
    r'^(\d+\.)+ .+': TypeReturn.LIST,
    r'^\* .+': TypeReturn.LIST,
    r'!\[.+\]\(.+\)': TypeReturn.IMAGE,
    r'\[\[\d+\]\]\(#footnote-\d+\)': TypeReturn.FOOTNOTE
}
multiline_types = {
    TypeReturn.TABLE,
    TypeReturn.LIST
}


class GraphParser:
    @staticmethod
    def get_type_return(s:  str) -> TypeReturn:
        """
        Определяет тип возвращаемого значения на основе входной строки.
        """
        for k, v in re2enum.items():
            if re.fullmatch(k, s) is not None:
                return v
        return TypeReturn.TEXT

    @staticmethod
    def to_imgs_base(
        images: list[ReaderImageOutput]
    ) -> dict[str, ReaderImageOutput]:
        """
        Преобразует список объектов ReaderImageOutput в словарь для 
        быстрого доступа по имени изображения.
        """
        return {
            img.image_name: img
            for img in images
        }

    @staticmethod
    def get_level(s: str | Image.Image) -> tuple[int, str | Image.Image]:
        """
        Определяет уровень заголовка или возвращает -1, если это изображение 
        или не заголовок.
        """
        if isinstance(s, Image.Image):
            return -1, s

        level_str = r'^#+ '
        match = re.match(level_str, s)
        if match is None:
            return -1, s

        return len(match.group().strip()), re.sub(level_str, '', s)

    @staticmethod
    def enum2value(graph: nx.Graph):
        graph = deepcopy(graph)
        new_data = {}
        for n in graph:
            new_data.update({
                n: graph.nodes[n]['node_type'].value
            })
        
        nx.set_node_attributes(graph, new_data, 'node_type')
        return graph
    
    @staticmethod
    def value2enum(graph: nx.Graph):
        graph = deepcopy(graph)
        new_data = {}
        for n in graph:
            new_data.update({
                n: TypeReturn(graph.nodes[n]['node_type'])
            })
        
        nx.set_node_attributes(graph, new_data, 'node_type')
        return graph

    def __init__(
        self,
        output_dir: str,
        gen_summary: Callable[[str], str],
        generate_summary=False
    ) -> None:
        """
        Инициализирует объект GraphParser.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.graph_file_path = os.path.join(output_dir, 'graph.pkl')

        self.gen_summary = gen_summary
        self.generate_summary = generate_summary

    def collate_f(
        self,
        reader: Iterable[str],
        imgs: dict[str, ReaderImageOutput]
    ) -> Iterable[GraphCollatorOutput]:
        """
        Сопоставляет входные данные с объектами GraphCollatorOutput, 
        обрабатывая изображения и многострочные типы.
        """
        last_type = None
        temp =  []
        for el in reader:
            el = el.strip()
            if el == '':
                continue

            type_ = GraphParser.get_type_return(el)

            if type_ == TypeReturn.IMAGE:
                img_text = re.findall(r'(?<=\[).+(?=\])', el)[0]
                el = el.replace(img_text, '')
                img_path = re.findall(r'(?<=\().+(?=\))', el)[0]

                yield GraphCollatorOutput(
                    type_,
                    img_text,
                    imgs[img_path].image,
                    summary=self.gen_summary(img_text) \
                        if self.generate_summary \
                        else None
                )
                continue

            if type_ in multiline_types:
                last_type = type_
                temp.append(el)
                continue

            if last_type is not None and last_type != type_:
                yield GraphCollatorOutput(
                    last_type,
                    '\n'.join(temp),
                    None,
                    summary=self.gen_summary('\n'.join(temp)) \
                        if self.generate_summary \
                        else None
                )
                temp = []
                last_type = None
            elif last_type is None:
                yield GraphCollatorOutput(
                    type_,
                    el,
                    None,
                    summary=self.gen_summary(el) \
                        if self.generate_summary \
                        else None
                )

    def text2graph(
        self,
        text: str,
        images: list[ReaderImageOutput],
        force_reload=False
    ) -> nx.Graph:
        """
        Преобразует входной текст и изображения в граф NetworkX.
        Если файл графа существует, загружает его. В противном случае 
        создает новый граф и сохраняет его.
        """
        if os.path.exists(self.graph_file_path) and not force_reload:
            return self.load_data()
        
        self.clear_saved_data()

        image_base = GraphParser.to_imgs_base(images)

        root_id = str(uuid4())
        last_seen_ids = [
            (root_id, -1)
        ]

        graph = nx.Graph()
        graph.add_node(
            root_id,
            position=-1,
            text='',
            node_type=TypeReturn.HEADING,
            image=None,
            level=-1
        )

        for i, data in enumerate(
            tqdm(
                self.collate_f(text.split('\n'), image_base),
                desc="Creating graph nodes"
            )
        ):
            current_level, line = GraphParser.get_level(data.text)

            node_id = str(uuid4())
            graph.add_node(
                node_id,
                position=i,
                text=line,
                node_type=data.obj_type,
                image=data.image,
                level=current_level \
                    if current_level != -1 \
                    else last_seen_ids[-1][1],
                summary=data.summary
            )

            while (
                current_level != -1
                and last_seen_ids[-1][1] >= current_level
            ):
                last_seen_ids.pop()
            
            graph.add_edge(last_seen_ids[-1][0], node_id)

            if current_level != -1:
                last_seen_ids.append((node_id, current_level))

        graph.remove_edges_from(nx.selfloop_edges(graph))

        self.save_data(graph)
    
        return graph
    
    def add_edges(self, graph: nx.Graph, window_size=6):
        """
        Добавляет ребра к графу на основе заданного размера окна.
        """
        with_ancestors = {
            i
            for i in graph.nodes
            if len(graph[i]) > 1
        }
        new_edge = []

        for id_ in tqdm(with_ancestors):
            nodes = list(graph[id_])
            for first_id, second_id in combinations(nodes, 2):
                distance = abs(
                    graph.nodes[first_id]['position'] 
                    - graph.nodes[second_id]['position']
                )
                if (
                    first_id in with_ancestors
                    or second_id in with_ancestors
                    or distance > window_size
                ):
                    continue
                new_edge.append((first_id, second_id))

        graph.add_edges_from(new_edge)
        graph.remove_edges_from(nx.selfloop_edges(graph))

        return graph
    
    def load_data(self):
        """
        Загружает сохранённый граф
        """
        with open(self.graph_file_path, 'rb') as f:
            graph = pickle.load(f)
        print(f'Граф загружен из {self.graph_file_path}')
        return GraphParser.value2enum(graph)

    def save_data(self, graph: nx.Graph):
        """
        Сохраняет данные графа в файл
        """
        os.makedirs(os.path.dirname(self.graph_file_path), exist_ok=True)
        with open(self.graph_file_path, 'wb') as f:
            pickle.dump(GraphParser.enum2value(graph), f)
        print(f'Граф сохранен в {self.graph_file_path}')

    def clear_saved_data(self):
        """
        Удаляет сохраненные файлы графа и эмбеддингов.
        """
        if os.path.exists(self.graph_file_path):
            os.remove(self.graph_file_path)
            print(f"Файл графа удален: {self.graph_file_path}")
