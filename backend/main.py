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
import re

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
    if not MISTRAL_API_KEY: return "Error: MISTRAL_API_KEY is not configured."
    try:
        llm = ChatMistralAI(model="mistral-large-latest", api_key=MISTRAL_API_KEY)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at extracting Ukrainian search keywords from a user question. Your goal is to pull out the most important terms for searching a government data portal. Respond with only the keywords, in Ukrainian, separated by spaces."),
            ("user", "{question}")
        ])
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"question": question})
    except Exception as e:
        print(f"Error in get_keywords_from_question: {e}")
        return "Error: Could not connect to AI model for keyword extraction."

def search_data_gov_ua(keywords: str) -> list:
    """Searches data.gov.ua and returns a list of dataset results or an error string."""
    if keywords.startswith("Error"): return keywords
    try:
        params = {'q': keywords, 'rows': 5}
        response = requests.get(DATA_GOV_UA_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("result", {}).get("results", [])
    except requests.exceptions.RequestException as e:
        print(f"Error in search_data_gov_ua: {e}")
        return f"Error: Failed to connect to data.gov.ua API: {e}"

def choose_best_dataset(question: str, search_results: list) -> dict | None:
    """Uses Mistral AI to choose the most relevant dataset from a list."""
    if not search_results: return None

    prompt_text = f"The user asked: '{question}'.\nI have found these datasets:\n"
    for i, dataset in enumerate(search_results):
        title = dataset.get('title', 'No title')
        notes = dataset.get('notes', 'No description available.')
        prompt_text += f"\n{i+1}. Title: {title}\n   Description: {notes[:200]}...\n" # Truncate description

    prompt_text += "\nWhich dataset is the most relevant to the user's question? Please respond with only the number of the best dataset."

    try:
        llm = ChatMistralAI(model="mistral-large-latest", api_key=MISTRAL_API_KEY)
        prompt = ChatPromptTemplate.from_messages([("user", "{prompt}")])
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"prompt": prompt_text})

        # Extract the first number from the response
        match = re.search(r'\d+', response)
        if match:
            choice_index = int(match.group(0)) - 1
            if 0 <= choice_index < len(search_results):
                return search_results[choice_index]
        return search_results[0] # Default to first result if parsing fails
    except Exception as e:
        print(f"Error in choose_best_dataset: {e}")
        return search_results[0] # Default to first result on error

# --- API Endpoints ---
@app.get("/", tags=["Status"])
async def read_root(): return {"status": "ok"}

@app.post("/api/query", tags=["AI Agent"], response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """The main endpoint for the AI agent."""
    print(f"Received question: {request.question}")

    keywords = get_keywords_from_question(request.question)
    print(f"Extracted keywords: {keywords}")

    search_results = search_data_gov_ua(keywords)
    if isinstance(search_results, str): # Check if an error string was returned
        return QueryResponse(answer=search_results)

    if not search_results:
        return QueryResponse(answer=f"I couldn't find any datasets for the keywords: '{keywords}'")

    # NEW: The reasoning step
    chosen_dataset = choose_best_dataset(request.question, search_results)
    print(f"Chosen dataset: {chosen_dataset.get('title') if chosen_dataset else 'None'}")

    if not chosen_dataset:
         answer = f"I found some datasets for '{keywords}', but couldn't determine the best one."
         source_url = None
    else:
        dataset_title = chosen_dataset.get('title', 'No title found')
        dataset_id = chosen_dataset.get('id')
        answer = f"Based on your question, the most relevant dataset I found is: '{dataset_title}'."
        source_url = f"https://data.gov.ua/dataset/{dataset_id}" if dataset_id else None

    return QueryResponse(answer=answer, source_url=source_url)

# --- Server Entrypoint ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
