import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# --- App Initialization ---
app = FastAPI(
    title="CivicData AI API",
    description="API for the AI agent that interacts with data.gov.ua.",
    version="0.1.0"
)

# --- CORS Configuration ---
# This allows the frontend (running on Netlify or locally) to communicate with this backend.
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8080",
    # Add your Netlify and OnRender frontend URLs here once you have them
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allowing all origins for now for easier deployment
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
    # **Placeholder Logic**
    # This is where we will build the core agent logic.
    # For now, it just echoes the question back.
    print(f"Received question: {request.question}")

    # 1. TODO: Extract keywords from the question.
    # 2. TODO: Search data.gov.ua using the keywords.
    # 3. TODO: Select the best dataset.
    # 4. TODO: Download and analyze the data with Pandas.
    # 5. TODO: Generate a natural language answer with Mistral AI.

    return QueryResponse(
        answer=f"This is a placeholder response. You asked: '{request.question}'",
        source_url="https://data.gov.ua",
        visualization=None
    )

# --- Server Entrypoint ---
if __name__ == "__main__":
    # This allows us to run the app with `python main.py` for local development.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)