import os
import re
import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import StringIO
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel

# --- Environment Variables ---
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# --- Constants ---
DATA_GOV_UA_API_URL = "https://data.gov.ua/api/3/action/package_search"

# --- App Initialization ---
app = FastAPI(
    title="CivicData AI API",
    description="API for the AI agent that interacts with data.gov.ua.",
    version="0.1.0",
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
    if not MISTRAL_API_KEY: return "Error: MISTRAL_API_KEY is not configured."
    try:
        llm = ChatMistralAI(model="mistral-large-latest", api_key=MISTRAL_API_KEY)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at extracting the 2-4 most essential Ukrainian search keywords from a user's question. Your goal is to pull out only the core terms needed to search a government data portal. Do not add extra words. Respond with only the keywords, in Ukrainian, separated by spaces. For example, if the user asks 'Скільки шкіл у місті Львів?', you should respond 'школи Львів'."),
            ("user", "{question}"),
        ])
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"question": question})
    except Exception as e:
        print(f"Error in get_keywords_from_question: {e}")
        return "Error: Could not connect to AI model for keyword extraction."

def search_data_gov_ua(keywords: str) -> list | str:
    if keywords.startswith("Error"): return keywords
    try:
        response = requests.get(DATA_GOV_UA_API_URL, params={'q': keywords, 'rows': 5})
        response.raise_for_status()
        return response.json().get("result", {}).get("results", [])
    except requests.exceptions.RequestException as e:
        print(f"Error in search_data_gov_ua: {e}")
        return f"Error: Failed to connect to data.gov.ua API: {e}"

def choose_best_dataset(question: str, search_results: list) -> dict | None:
    if not search_results: return None
    prompt_text = f"The user asked: '{question}'.\nI have found these datasets:\n"
    for i, dataset in enumerate(search_results):
        prompt_text += f"\n{i+1}. Title: {dataset.get('title', 'No title')}\n   Description: {dataset.get('notes', 'No description')[:200]}...\n"
    prompt_text += "\nWhich dataset is most relevant? Respond with only the number."
    try:
        llm = ChatMistralAI(model="mistral-large-latest", api_key=MISTRAL_API_KEY)
        chain = ChatPromptTemplate.from_messages([("user", "{prompt}")]) | llm | StrOutputParser()
        response = chain.invoke({"prompt": prompt_text})
        match = re.search(r'\d+', response)
        if match:
            choice_index = int(match.group(0)) - 1
            if 0 <= choice_index < len(search_results):
                return search_results[choice_index]
        return search_results[0]
    except Exception as e:
        print(f"Error in choose_best_dataset: {e}")
        return search_results[0]

def analyze_data_with_ai(question: str, data_file_url: str) -> str:
    """Downloads a data file, loads it into a pandas DataFrame, and uses a LangChain agent to answer a question."""
    if not MISTRAL_API_KEY:
        return "Error: MISTRAL_API_KEY is not configured."
    try:
        # Download the data file
        response = requests.get(data_file_url)
        response.raise_for_status()
        
        # Attempt to decode with UTF-8, then fall back to 'cp1251' for older Ukrainian files
        try:
            data = response.content.decode('utf-8')
        except UnicodeDecodeError:
            data = response.content.decode('cp1251')

        # Use StringIO to handle the string data as a file and create a DataFrame
        df = pd.read_csv(StringIO(data))
        
        # Initialize the AI model
        llm = ChatMistralAI(model="mistral-large-latest", api_key=MISTRAL_API_KEY, temperature=0)
        
        # Create the Pandas DataFrame agent
        agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
        
        # Ask the agent the user's question
        # The agent will intelligently write and execute python code to answer the question
        answer = agent.invoke(question)
        
        return answer['output']

    except requests.exceptions.RequestException as e:
        return f"Error: Could not download the data file from the provided URL. {e}"
    except Exception as e:
        return f"Error: Failed to analyze the data. The file might not be a valid CSV or there was another issue. {e}"

# --- API Endpoints ---
@app.get("/", tags=["Status"])
async def read_root(): return {"status": "ok"}

@app.post("/api/query", tags=["AI Agent"], response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    print(f"Received question: {request.question}")
    keywords = get_keywords_from_question(request.question)
    print(f"Extracted keywords: {keywords}")
    search_results = search_data_gov_ua(keywords)
    if isinstance(search_results, str):
        return QueryResponse(answer=search_results)
    if not search_results:
        return QueryResponse(answer=f"I couldn't find any datasets for: '{keywords}'")

    chosen_dataset = choose_best_dataset(request.question, search_results)
    if not chosen_dataset:
        return QueryResponse(answer=f"I found datasets for '{keywords}', but couldn't choose the best one.")
    
    dataset_title = chosen_dataset.get('title', 'No title')
    print(f"Chosen dataset: {dataset_title}")

    # NEW: Find the actual data file URL
    data_file_url = find_data_file_url(chosen_dataset)
    print(f"Found data file URL: {data_file_url}")

    if not data_file_url:
        answer = f"I found the dataset '{dataset_title}', but couldn't find a downloadable data file inside it."
        source_url = f"https://data.gov.ua/dataset/{chosen_dataset.get('id')}"
    else:
        # Now, analyze the data to get the final answer
        print("Analyzing data file...")
        ai_analysis = analyze_data_with_ai(request.question, data_file_url)
        answer = ai_analysis
        source_url = data_file_url # Return the direct file URL

    return QueryResponse(answer=answer, source_url=source_url)

# --- Server Entrypoint ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

def find_data_file_url(dataset: dict) -> str | None:
    """Finds a suitable data file URL (CSV, Excel) from the dataset resources."""
    resources = dataset.get('resources', [])
    if not resources:
        return None
        
    # Priority formats for analysis
    target_formats = ['CSV', 'XLSX', 'XLS']
    
    for fmt in target_formats:
        for resource in resources:
            if resource.get('format', '').upper() == fmt:
                return resource.get('url')
    
    # Fallback: return the first resource URL if no specific format matches
    if resources:
        return resources[0].get('url')
    
    return None
