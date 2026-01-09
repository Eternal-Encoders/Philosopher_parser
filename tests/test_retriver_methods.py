import os

import networkx as nx
import numpy as np
import pytest

from src.graphs import Retriver, TypeReturn

test_payload = {
    'f_path': '__input_TEST__/test.docx',
    'root_dir': '__output_TEST__',
    'encode_q_fn': lambda texts: np.random.rand(len(texts), 768),
    'encode_d_fn': lambda texts: np.random.rand(len(texts), 768),
    'ocr_fn': lambda images: ['ocr text'] * len(images),
    'gen_summary_fn': lambda text: f'Summary of {text[:20]}',
}


@pytest.fixture
def mock_graph():
    graph = nx.Graph()
    graph.add_node(
        'node1',
        text='Text 1',
        level=1,
        node_type=TypeReturn.TEXT)
    graph.add_node(
        'node2',
        text='Text 2',
        level=1,
        node_type=TypeReturn.HEADING)
    graph.add_node(
        'node3',
        text='Text 3',
        level=2,
        node_type=TypeReturn.TEXT)
    graph.add_edge('node1', 'node2')
    graph.add_edge('node2', 'node3')

    return graph


@pytest.fixture
def mock_retriver_t():
    return Retriver(
        force_reload=True,
        **test_payload
    )


@pytest.fixture
def mock_retriver_f():
    return Retriver(
        force_reload=False,
        **test_payload
    )


def test_get_ids_text(mock_graph):
    texts, ids = Retriver.get_ids_text(mock_graph, text_field='text')
    
    assert isinstance(texts, list)
    assert isinstance(ids, np.ndarray)
    assert len(texts) == len(ids)
    assert len(texts) == 3
    assert 'Text 1' in texts
    assert 'Text 2' in texts
    assert 'Text 3' in texts


def test_prepare_data_new(mock_retriver_t: Retriver):
    emb_path = os.path.join(mock_retriver_t.root_dir, 'binaries', 'docs.pkl')
    ids_path = os.path.join(mock_retriver_t.root_dir, 'binaries', 'ids.pkl')

    assert isinstance(mock_retriver_t.doc_emb, np.ndarray)
    assert isinstance(mock_retriver_t.node_ids, np.ndarray)
    assert len(mock_retriver_t.doc_emb) == len(mock_retriver_t.node_ids)
    assert os.path.exists(emb_path)
    assert os.path.exists(ids_path)


def test_clear_data(mock_retriver_t: Retriver):
    emb_path = os.path.join(mock_retriver_t.root_dir, 'binaries', 'docs.pkl')
    ids_path = os.path.join(mock_retriver_t.root_dir, 'binaries', 'ids.pkl')
    
    mock_retriver_t.clear_data()

    assert not os.path.exists(emb_path)
    assert not os.path.exists(ids_path)


def test_get_similar(mock_retriver_t: Retriver):        
    query = np.random.rand(768)
    similar = mock_retriver_t.get_similar(query, top_k=2)
    
    assert isinstance(similar, np.ndarray)
    assert len(similar) <= 2


def test_retrive_docs_string_query(mock_retriver_t: Retriver):        
    docs = mock_retriver_t.retrive_docs('test query', top_k=2)

    assert isinstance(docs, list)
    assert len(docs) <= 2


# def test_retrive_docs_list_query(mock_retriver_t: Retriver):
#     docs = mock_retriver_t.retrive_docs(['query1', 'query2'], top_k=2)

#     assert isinstance(docs, list)


def test_retrive_docs_array_query(mock_retriver_t: Retriver):
    query = np.random.rand(768)
    docs = mock_retriver_t.retrive_docs(query, top_k=2)

    assert isinstance(docs, list)


def test_get_questions(mock_retriver_t: Retriver):
    questions = mock_retriver_t.get_questions()

    assert isinstance(questions, set)
