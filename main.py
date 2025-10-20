from fastapi import FastAPI
from src import GraphRetriver, Connector, RAGModel

app = FastAPI()

parser = GraphRetriver()
parser.load_graph('./__output__/binaries')

conn = Connector(parser)


@app.get('/')
async def root():
    return 'ok'


@app.post('/rag')
async def rag(data: RAGModel):
    res = conn.get_docs(data.query)
    return res