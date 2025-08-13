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
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# --- Constants ---
DATA_GOV_UA_API_URL = "https://data.gov.ua/api/3/action/package_search"

# --- App Initialization ---
app = FastAPI(
    title="CivicData AI API",
    description="API for the AI agent that interacts with data.gov.ua.",
    version="0.1.0"
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

# --- AI & Data Functions ---
def get_keywords_from_question(question: str) -> str:
    """Uses Mistral AI to extract search keywords from a user's question."""
    if not MISTRAL_API_KEY:
        return "Error: MISTRAL_API_KEY is not configured."
    try:
        llm = ChatMistralAI(model="mistral-large-latest", api_key=MISTRAL_API_KEY)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at extracting Ukrainian search keywords from a user question. Your goal is to pull out the most important terms for searching a government data portal. Respond with only the keywords, in Ukrainian, separated by spaces."),
            ("user", "{question}")
        ])
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        return chain.invoke({"question": question})
    except Exception as e:
        print(f"Error in get_keywords_from_question: {e}")
        return "Error: Could not connect to AI model."

def search_data_gov_ua(keywords: str) -> dict:
    """Searches data.gov.ua for datasets matching the keywords."""
    if keywords.startswith("Error"):
        return {"error": keywords}
    try:
        params = {'q': keywords, 'rows': 5} # Ask for 5 results
        response = requests.get(DATA_GOV_UA_API_URL, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        if data.get("result", {}).get("results"):
            return data["result"]["results"]
        else:
            return {"error": "No datasets found for these keywords."}
    except requests.exceptions.RequestException as e:
        print(f"Error in search_data_gov_ua: {e}")
        return {"error": f"Failed to connect to data.gov.ua API: {e}"}

# --- API Endpoints ---
@app.get("/", tags=["Status"])
async def read_root():
    return {"status": "ok", "message": "Welcome to the CivicData AI API!"}

@app.post("/api/query", tags=["AI Agent"], response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """The main endpoint for the AI agent."""
    print(f"Received question: {request.question}")

    # Step 1: Extract keywords from the question.
    keywords = get_keywords_from_question(request.question)
    print(f"Extracted keywords: {keywords}")

    # Step 2: Search data.gov.ua using the keywords.
    search_results = search_data_gov_ua(keywords)
    print(f"Search results: {search_results}")

    # TODO: Step 3: Select the best dataset from the results.
    # TODO: Step 4: Download and analyze the data.
    # TODO: Step 5: Generate a natural language answer.

    # For now, return the name of the first found dataset to test this step.
    if "error" in search_results:
        answer = search_results["error"]
        source_url = None
    elif not search_results:
         answer = f"I couldn't find any datasets for the keywords: '{keywords}'"
         source_url = None
    else:
        first_dataset = search_results[0]
        dataset_title = first_dataset.get('title', 'No title found')
        dataset_id = first_dataset.get('id')
        answer = f"I searched for '{keywords}' and found this dataset: '{dataset_title}'."
        source_url = f"https://data.gov.ua/dataset/{dataset_id}" if dataset_id else None

    return QueryResponse(answer=answer, source_url=source_url)

# --- Server Entrypoint ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)