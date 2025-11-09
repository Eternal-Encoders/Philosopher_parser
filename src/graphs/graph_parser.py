import re
import pickle
import networkx as nx
import os
from dotenv import load_dotenv
from uuid import uuid4
from PIL import Image
from typing import Iterable, List, Dict
from .models import ReaderImageOutput, GraphCollatorOutput
from ..models import TypeReturn
from itertools import combinations
from tqdm import tqdm
from openai import OpenAI

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


def __get_type_return(s:  str) -> TypeReturn:
    """
    Определяет тип возвращаемого значения на основе входной строки.
    """
    n = re.sub(r'(^[\*]+)|([\*]+$)', '', s)
    for k, v in re2enum.items():
        if re.fullmatch(k, n) is not None:
            return v
    return TypeReturn.TEXT


def collate_f(reader: Iterable[str], imgs: Dict[str, ReaderImageOutput]) -> Iterable[GraphCollatorOutput]:
    """
    Сопоставляет входные данные с объектами GraphCollatorOutput, обрабатывая изображения и многострочные типы.
    """
    last_type = None
    temp =  []
    for el in reader:
        el = el.strip()
        if el == '':
            continue

        type_ = __get_type_return(el)

        if type_ == TypeReturn.IMAGE:
            img_text = re.findall(r'(?<=\[).+(?=\])', el)[0]
            el = el.replace(img_text, '')
            img_path = re.findall(r'(?<=\().+(?=\))', el)[0]

            yield GraphCollatorOutput(
                type_,
                img_text,
                imgs[img_path].image,
                summary=generate_summary_with_llm(img_text)
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
                summary=generate_summary_with_llm('\n'.join(temp))
            )
            temp = []
            last_type = None
        elif last_type is None:
            yield GraphCollatorOutput(
                type_,
                el,
                None,
                summary=generate_summary_with_llm(el)
            )


def to_imgs_base(images: List[ReaderImageOutput]) -> Dict[str, ReaderImageOutput]:
    """
    Преобразует список объектов ReaderImageOutput в словарь для быстрого доступа по имени изображения.
    """
    return {
        'media/' + img.image_name: img
        for img in images
    }


def get_level(s: str | Image.Image) -> tuple[int, str | Image.Image]:
    """
    Определяет уровень заголовка или возвращает -1, если это изображение или не заголовок.
    """
    if isinstance(s, Image.Image):
        return -1, s

    level_str = r'^#+ '
    match = re.match(level_str, s)
    if match is None:
        return -1, s

    return len(match.group().strip()), re.sub(level_str, '', s)


class GraphParser:
    def __init__(self, output_dir: str) -> None:
        """
        Инициализирует объект GraphParser.
        """
        self.output_dir = output_dir
        self.graph_file_path = os.path.join(output_dir, 'graph.pkl')
        self.nodes_data_file_path = os.path.join(output_dir, 'nodes_data.pkl')

    def text2graph(self, text: str, images: List[ReaderImageOutput]):
        """
        Преобразует входной текст и изображения в граф NetworkX.
        Если файл графа существует, загружает его. В противном случае создает новый граф и сохраняет его.
        """
        if os.path.exists(self.graph_file_path):
            with open(self.graph_file_path, 'rb') as f:
                graph = pickle.load(f)
            print(f"Граф загружен из {self.graph_file_path}")
            return graph

        image_base = to_imgs_base(images)

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

        for i, data in enumerate(tqdm(collate_f(text.split('\n'), image_base), desc="Creating graph nodes")):
            current_level, line = get_level(data.text)

            node_id = str(uuid4())
            graph.add_node(
                node_id,
                position=i,
                text=line,
                node_type=data.obj_type,
                image=data.image,
                level=current_level if current_level != -1 else last_seen_ids[-1][1],
                summary=data.summary
            )

            while current_level != -1 and last_seen_ids[-1][1] >= current_level:
                last_seen_ids.pop()
            
            graph.add_edge(last_seen_ids[-1][0], node_id)

            if current_level != -1:
                last_seen_ids.append((node_id, current_level))

        graph.remove_edges_from(nx.selfloop_edges(graph))
        
        # Сохранение графа после создания
        os.makedirs(os.path.dirname(self.graph_file_path), exist_ok=True)
        with open(self.graph_file_path, 'wb') as f:
            pickle.dump(graph, f)
        print(f"Граф сохранен в {self.graph_file_path}")

        with open(self.nodes_data_file_path, 'wb') as f:
            pickle.dump(
                [
                    {
                        'id': k,
                        **v
                    }
                    for k, v in dict(graph.nodes.data()).items()
                ],
                f
            )
        print(f"Данные узлов сохранены в {self.nodes_data_file_path}")
    
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
                distance = abs(graph.nodes[first_id]['position'] - graph.nodes[second_id]['position'])
                if first_id in with_ancestors or second_id in with_ancestors or distance > window_size:
                    continue
                new_edge.append((first_id, second_id))

        graph.add_edges_from(new_edge)
        graph.remove_edges_from(nx.selfloop_edges(graph))

        return graph
    
    def clear_saved_data(self):
        """
        Удаляет сохраненные файлы графа и эмбеддингов.
        """
        if os.path.exists(self.graph_file_path):
            os.remove(self.graph_file_path)
            print(f"Файл графа удален: {self.graph_file_path}")
        if os.path.exists(self.nodes_data_file_path):
            os.remove(self.nodes_data_file_path)
            print(f"Файл данных узлов удален: {self.nodes_data_file_path}")

        # Также нужно удалить файлы эмбеддингов, которые сохраняются в retriver.py
        # Предполагаем, что они находятся в той же директории, что и GRAPH_FILE_PATH
        emb_path = os.path.join(self.output_dir, 'docs.pkl')
        ids_path = os.path.join(self.output_dir, 'ids.pkl')

        if os.path.exists(emb_path):
            os.remove(emb_path)
            print(f"Файл эмбеддингов удален: {emb_path}")
        if os.path.exists(ids_path):
            os.remove(ids_path)
            print(f"Файл ID узлов удален: {ids_path}")


load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def generate_summary_with_llm(text: str) -> str:
    """
    Генерирует краткое саммари текста с помощью LLM через OpenRouter.
    """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    try:
        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct",  # Можно выбрать другую модель
            messages=[
                {"role": "user", "content": f"Сделай краткое саммари следующего текста: {text}"}
            ],
            temperature=0.7,
            max_tokens=200,
        )
        summary_content = response.choices[0].message.content
        return summary_content.strip() if summary_content else text[:100] + "..." if len(text) > 100 else text
    except Exception as e:
        print(f"Ошибка при генерации саммари с помощью LLM: {e}")
        return text[:100] + "..." if len(text) > 100 else text