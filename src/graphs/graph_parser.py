import re
import pickle
import networkx as nx
from uuid import uuid4
from PIL import Image
from typing import Iterable, List, Dict
from .models import ReaderImageOutput, GraphCollatorOutput
from ..models import TypeReturn
from itertools import combinations
from tqdm.notebook import tqdm

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
    n = re.sub(r'(^[\*]+)|([\*]+$)', '', s)
    for k, v in re2enum.items():
        if re.fullmatch(k, n) is not None:
            return v
    return TypeReturn.TEXT


def collate_f(reader: Iterable[str], imgs: Dict[str, ReaderImageOutput]) -> Iterable[GraphCollatorOutput]:
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
                imgs[img_path].image
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
                None
            )
            temp = []
            last_type = None
        elif last_type is None:
            yield GraphCollatorOutput(
                type_,
                el,
                None
            )


def to_imgs_base(images: List[ReaderImageOutput]) -> Dict[str, ReaderImageOutput]:
    return {
        img.image_name: img
        for img in images
    }


def get_level(s: str | Image.Image) -> tuple[int, str | Image.Image]:
    if isinstance(s, Image.Image):
        return -1, s

    level_str = r'^#+ '
    match = re.match(level_str, s)
    if match is None:
        return -1, s

    return len(match.group().strip()), re.sub(level_str, '', s)


class GraphParser:
    def __init__(self) -> None:
        pass

    def text2graph(self, text: str, images: List[ReaderImageOutput]):
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

        for i, data in enumerate(collate_f(text.split('\n'), image_base)):
            current_level, line = get_level(data.text)

            node_id = str(uuid4())
            graph.add_node(
                node_id,
                position=i,
                text=line,
                node_type=data.obj_type,
                image=data.image,
                level=current_level if current_level != -1 else last_seen_ids[-1][1]
            )

            while current_level != -1 and last_seen_ids[-1][1] >= current_level:
                last_seen_ids.pop()
            
            graph.add_edge(last_seen_ids[-1][0], node_id)

            if current_level != -1:
                last_seen_ids.append((node_id, current_level))

        graph.remove_edges_from(nx.selfloop_edges(graph))
    
        return graph
    
    def add_edges(self, graph: nx.Graph, window_size=6):
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


# with open('./__output__/binaries/graph_1.pkl', 'wb') as f:
#     pickle.dump(G, f)

# with open('./__output__/binaries/nodes_data.pkl', 'wb') as f:
#     pickle.dump(
#         [
#             {
#                 'id': k,
#                 **v
#             }
#             for k, v in dict(G.nodes.data()).items()
#         ],
#         f
#     )