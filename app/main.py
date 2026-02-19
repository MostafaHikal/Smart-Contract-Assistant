import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from app.retrieval import get_contract_response
import uvicorn

load_dotenv()

app = FastAPI(
    title="Smart Contract Assistant API",
    version="1.0",
    description="Backend server for contract analysis using RAG"
)

# 1. Define Request Body
class Query(BaseModel):
    question: str

# 2. Root Endpoint
@app.get("/")
async def root():
    return {"message": "Smart Contract API is running!"}

# 3. Ask Endpoint (The Bridge to Retrieval)
@app.post("/ask")
async def ask_contract(query: Query):
    answer = get_contract_response(query.question)
    return {
        "question": query.question, 
        "answer": answer
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)   