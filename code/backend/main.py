from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from retrieve import Retrieval
from synthesize import Synthesis
import os
import warnings
warnings.filterwarnings("ignore")
from paths import ENV_FPATH

from dotenv import load_dotenv
load_dotenv(dotenv_path=ENV_FPATH)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

# class QueryResponse(BaseModel):
#     response: str
class QueryResponse(BaseModel):
    response: str
    images: list[str] = []
    tables: list[str] = []

# Initialize the retriever
retriever = Retrieval()
synthesizer = Synthesis(groq_api_key=os.getenv("GROQ_API_KEY"))

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        relevent_docs = retriever.retrieve(request.query)
        synthesizer.configure_prompt_settings(relevant_docs=relevent_docs, retrieval=retriever)
        answer, images, tables = synthesizer.get_llm_response(request.query)
        return QueryResponse(response=answer, images=images, tables=tables)
    except Exception as e:
        return QueryResponse(response=f"Error occurred: {str(e)}")
