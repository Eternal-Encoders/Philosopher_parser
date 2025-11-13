import os
import dotenv
from fastapi import FastAPI
from src import Retriver, Connector, RAGModel
from src.model_inf import VectorizerExec, OcrExec

dotenv.load_dotenv()
app = FastAPI()

ocr = OcrExec(
    os.environ['OPENROUTER_API_KEY'],
    os.environ['OCR_MODEL']
)
vectorizer = VectorizerExec(
    os.environ['HUGGINGFACE_HUB_TOKEN'],
    os.environ['VECTORIZER_MODEL']
)

parser = Retriver(
    encode_q_fn=vectorizer.encode,
    encode_d_fn=vectorizer.encode,
    ocr_fn=ocr.ocr
)
parser.load_graph('/__output__/binaries')

conn = Connector(parser)


@app.get('/')
async def root():
    return 'ok'


@app.post('/rag')
async def rag(data: RAGModel):
    res = conn.get_docs(data.query)
    return res


@app.get('/questions')
async def questions():
    return conn.get_questions()
