import os
import networkx as nx
from dotenv import load_dotenv
from src.graphs import FileReader, GraphParser
from src.model_inf import OcrExec, SummaryExec

load_dotenv()

ocr = OcrExec(
    os.environ['OPENROUTER_API_KEY'],
    os.environ['OCR_MODEL']
)
summary_gen = SummaryExec(
    os.environ['OPENROUTER_API_KEY'],
    os.environ['SUMMARY_MODEL']
)
file_reader = FileReader(ocr.ocr, root_path='__output_TEST__')
doc, images = file_reader.read_markdown('__input_TEST__/test.docx', force_reload=False)

graph_parser = GraphParser('__output_TEST__/binaries', summary_gen.generate_summary_text)

def test_parsing():
    graph = graph_parser.text2graph(doc, images, force_reload=True)

    assert isinstance(graph, nx.Graph)

def test_graph_load():
    graph = graph_parser.text2graph(doc, images, force_reload=False)
    graph = graph_parser.text2graph(doc, images, force_reload=False)

    assert isinstance(graph, nx.Graph)