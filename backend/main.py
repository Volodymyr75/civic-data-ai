import os
import re
import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import StringIO, BytesIO
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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
        response = requests.get(DATA_GOV_UA_API_URL, params={'q': keywords, 'rows': 20})
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
        # Download the data file with a User-Agent and SSL verify=False to avoid 403/SSL errors
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "uk-UA,uk;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://data.gov.ua/"
        }
        # Suppress InsecureRequestWarning since we are deliberately disabling verify
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        response = requests.get(data_file_url, headers=headers, verify=False)
        response.raise_for_status()
        
        # Determine file type and load accordingly
        if data_file_url.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(BytesIO(response.content))
        else:
            # Attempt to decode with UTF-8, then fall back to 'cp1251' for older Ukrainian files
            try:
                data = response.content.decode('utf-8')
            except UnicodeDecodeError:
                data = response.content.decode('cp1251')
            # Use StringIO to handle the string data as a file
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
        raise Exception(f"Could not download the data file: {e}") # Re-raise to be caught by retry loop
    except Exception as e:
        raise Exception(f"Failed to analyze the data: {e}") # Re-raise to be caught by retry loop

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

    # NEW: Filter datasets to only include those with valid data files (CSV, XLSX, XLS)
    valid_datasets = []
    for dataset in search_results:
        if find_data_file_url(dataset):
            valid_datasets.append(dataset)
    
    print(f"Found {len(search_results)} total datasets, {len(valid_datasets)} with valid data files.")

    # If we have valid datasets, use them. Otherwise, fall back to all results (better than nothing)
    datasets_to_choose_from = valid_datasets if valid_datasets else search_results

    # RETRY LOOP: Try up to 3 datasets
    max_retries = 3
    attempts = 0
    
    while attempts < max_retries and datasets_to_choose_from:
        attempts += 1
        print(f"Attempt {attempts}/{max_retries} to choose and analyze a dataset...")
        
        chosen_dataset = choose_best_dataset(request.question, datasets_to_choose_from)
        if not chosen_dataset:
             return QueryResponse(answer=f"I found datasets for '{keywords}', but couldn't choose the best one.")
        
        dataset_title = chosen_dataset.get('title', 'No title')
        print(f"Chosen dataset: {dataset_title}")

        # Find the actual data file URL
        data_file_url = find_data_file_url(chosen_dataset)
        print(f"Found data file URL: {data_file_url}")

        if not data_file_url:
            # This shouldn't happen often if valid_datasets logic is working, but just in case
            print("No valid file URL found in chosen dataset. Skipping.")
            datasets_to_choose_from.remove(chosen_dataset)
            continue

        try:
            # Now, analyze the data to get the final answer
            print("Analyzing data file...")
            ai_analysis = analyze_data_with_ai(request.question, data_file_url)
            return QueryResponse(answer=ai_analysis, source_url=data_file_url)
        except Exception as e:
            print(f"Error analyzing dataset '{dataset_title}': {e}")
            # Remove the failed dataset from the list and try again
            if chosen_dataset in datasets_to_choose_from:
                datasets_to_choose_from.remove(chosen_dataset)
            
            if not datasets_to_choose_from:
                 return QueryResponse(answer=f"I tried to analyze several datasets for '{keywords}', but encountered errors (e.g., access denied or invalid format). Last error: {e}")

    return QueryResponse(answer=f"I couldn't successfully analyze any of the found datasets for '{keywords}'.")

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
    
    # STRICT MODE: Do NOT fallback to just any file. 
    # If we can't find a CSV or Excel, we can't analyze it reliably.
    return None
