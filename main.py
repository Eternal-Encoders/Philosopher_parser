import os
import dotenv
import uvicorn

from fastapi import FastAPI
from src import Retriver, RAGModel
from fastapi.responses import FileResponse
from src.model_inf import VectorizerExec, OcrExec, SummaryExec

dotenv.load_dotenv()
app = FastAPI()

ocr = OcrExec(
    os.environ['OCR_ENDPOINT'],
    os.environ['OPENROUTER_API_KEY'],
    os.environ['OCR_MODEL']
)
vectorizer = VectorizerExec(
    os.environ['VECTORIZER_ENDPOINT'],
    os.environ['HUGGINGFACE_HUB_TOKEN'],
    os.environ['VECTORIZER_MODEL']
)
summary_gen = SummaryExec(
    os.environ['SUMMARY_ENDPOINT'],
    os.environ['OPENROUTER_API_KEY'],
    os.environ['SUMMARY_MODEL']
)

parser = Retriver(
    f_path='__input__/Учебник_философии_22_августа_ТюмГУ.docx',
    encode_q_fn=vectorizer.encode,
    encode_d_fn=vectorizer.encode,
    ocr_fn=ocr.ocr,
    gen_summary_fn=summary_gen.generate_summary_text,
    text_field='text',
    root_dir='__output__',
    force_reload=False
)


@app.get('/')
async def root():
    return 'ok'


@app.post('/rag')
async def rag(data: RAGModel):
    res = parser.retrive_docs(data.query)
    return res


@app.get('/questions')
async def questions():
    return parser.get_questions()

@app.get('/document')
async def document():
    return FileResponse(parser.file_reader.md_path)


if __name__ == '__main__':
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=80,
        workers=1
    )