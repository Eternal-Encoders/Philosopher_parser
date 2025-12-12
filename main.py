import os
import dotenv
from typing import List, Optional
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from src import Retriver, RAGModel
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


# --------------------
# Response Models
# --------------------
class StatusResponse(BaseModel):
    status: str


class RagResponse(BaseModel):
    docs: List[str]
    meta: Optional[dict] = None


class QuestionItem(BaseModel):
    text: str


@app.get('/', response_model=StatusResponse, tags=["health"])
async def root() -> StatusResponse:
    return StatusResponse(status='ok')


@app.post('/rag', response_model=RagResponse, tags=["rag"])
async def rag(data: RAGModel) -> RagResponse:
    docs = parser.retrive_docs(query=data.query, top_k=data.top_k)  # List[str]
    res_docs = []
    len_docs = 0
    i = 0
    while i < len(docs):
        if len_docs <= data.max_length:
            res_docs.append(docs[i])
            len_docs += len(docs[i])
            i += 1
        else:
            break
    return RagResponse(docs=res_docs, meta=None)

@app.get("/health")
async def health_check():
    """Проверка здоровья сервера"""
    return {"status": "healthy", "service": "philosopher-rag-api"}

@app.get('/questions', response_model=List[QuestionItem], tags=["rag"])
async def questions() -> List[QuestionItem]:
    q = parser.get_questions()  # set[str]
    return [QuestionItem(text=str(it)) for it in (q or [])]

@app.get('/document', tags=["rag"])
async def document():
    path = parser.file_reader.md_path
    filename = os.path.basename(path)
    return FileResponse(path, media_type="text/markdown", filename=filename)
