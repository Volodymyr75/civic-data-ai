import os
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Environment Variables ---
# Load environment variables from .env file for local development
load_dotenv()
# OnRender will set this environment variable automatically
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# --- App Initialization ---
app = FastAPI(
    title="CivicData AI API",
    description="API for the AI agent that interacts with data.gov.ua.",
    version="0.1.0"
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for simplicity, can be restricted later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    source_url: str | None = None
    visualization: dict | None = None

# --- AI Keyword Extraction Logic ---
def get_keywords_from_question(question: str) -> str:
    """
    Uses Mistral AI to extract relevant search keywords from a user's question.
    """
    if not MISTRAL_API_KEY:
        return "Error: MISTRAL_API_KEY is not configured on the server."

    try:
        llm = ChatMistralAI(model="mistral-large-latest", api_key=MISTRAL_API_KEY)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at extracting Ukrainian search keywords from a user question. Your goal is to pull out the most important terms that would be used to search a government open data portal. Respond with only the keywords, in Ukrainian, separated by spaces. Do not use any other words or punctuation."),
            ("user", "{question}")
        ])
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        keywords = chain.invoke({"question": question})
        return keywords
    except Exception as e:
        print(f"Error during keyword extraction: {e}")
        return f"Error: Could not connect to AI model. Please check the API key and model configuration."

# --- API Endpoints ---
@app.get("/", tags=["Status"])
async def read_root():
    """A simple endpoint to check if the API is running."""
    return {"status": "ok", "message": "Welcome to the CivicData AI API!"}

@app.post("/api/query", tags=["AI Agent"], response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """
    The main endpoint for the AI agent.
    It takes a user's question, finds relevant data, analyzes it, and returns an answer.
    """
    print(f"Received question: {request.question}")

    # 1. Extract keywords from the question using Mistral AI.
    keywords = get_keywords_from_question(request.question)
    print(f"Extracted keywords: {keywords}")

    # 2. TODO: Search data.gov.ua using the keywords.
    # 3. TODO: Select the best dataset.
    # 4. TODO: Download and analyze the data with Pandas.
    # 5. TODO: Generate a natural language answer with Mistral AI.

    # For now, we return the extracted keywords to test this step.
    return QueryResponse(
        answer=f"Extracted keywords: {keywords}",
        source_url=None,
        visualization=None
    )

# --- Server Entrypoint ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
