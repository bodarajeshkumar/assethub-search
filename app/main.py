from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from milvus_wrapper import MilvusWrapper
import json
import os
from dotenv import load_dotenv
from ibm_watson_machine_learning.foundation_models import Model
import re
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Define default parameters for LLM
DEFAULT_LLM_PARAMS = {
    'decoding_method': "greedy",
    'min_new_tokens': 1,
    'max_new_tokens': 400,
    'repetition_penalty': 1,
    'random_seed': 42,
}

# Prompt template
DEFAULT_PROMPT = '''
You are a helpful, respectful, and honest assistant. You will be provided with a set of search results and a user query. 
Use the search results to answer the user's query as accurately as possible. Make sure to include the relevant title 
and author name from the search results in your response. However, if the user query already mentions an author, 
do not include that author's name in the response.
Ensure that your answer is clear, concise, and based only on the provided search results. If the query is not related to 
the search results or cannot be answered using the given information, kindly inform the user that the relevant information 
is not available.
User query: {user_query_text}
Search results:
{search_results}
I need ID in the answer wrt the search.
Provide the most relevant answer, including the ID, title, author name and description. 
'''

# LLM Backend Cloud class
class LLMBackendCloud():
    def __init__(self, model_id='meta-llama/llama-3-70b-instruct', model_params=DEFAULT_LLM_PARAMS):
        api_key = os.getenv("API_KEY", None)
        ibm_cloud_url = os.getenv("API_ENDPOINT", None)
        project_id = os.getenv("PROJECT_ID", None)
        if api_key is None or ibm_cloud_url is None or project_id is None:
            raise ValueError("Ensure the .env file is in place with correct API details")
        self.creds = {
            "url": ibm_cloud_url,
            "apikey": api_key
        }
        self.model_id = model_id
        self.model_params = model_params
        self.model = Model(model_id=self.model_id, params=self.model_params, credentials=self.creds, project_id=project_id)

    def generate_response(self, prompt: str, model_params=None):
        if not model_params:
            model_params = self.model_params
        result = self.model.generate_text(prompt=prompt, params=model_params)
        return result

# Request body model
class QueryRequest(BaseModel):
    user_query_text: str

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utility functions
def extract_author_from_query(query):
    match = re.search(r'by ([A-Za-z ]+)', query, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def extract_entities_from_query(query):
    tech_keywords = ['wxo', 'orchestrate', 'watson orchestrate', 'wxa', 'assistant', 'watson assistant']
    entities = []
    query = re.sub(r'\blist\b|\ball\b|\bfor\b|\bassets\b', '', query, flags=re.IGNORECASE).strip()
    for keyword in tech_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', query, re.IGNORECASE):
            entities.append(keyword)
    if not entities:
        search_terms = re.findall(r'\b([a-zA-Z\s]+)', query)
        for term in search_terms:
            term = term.strip()
            if len(term) > 2:
                entities.append(term)
    return entities if entities else None

def extract_relevant_keywords(query):
    query = query.lower()
    author = extract_author_from_query(query)
    entities = extract_entities_from_query(query)
    return {'author': author, 'entities': entities}

def prepare_prompt(user_query_text, search_results):
    return DEFAULT_PROMPT.format(user_query_text=user_query_text, search_results=search_results)

def map_hashed_ids_to_original(milvus_wrapper, search_results):
    for result in search_results:
        original_id = milvus_wrapper.id_mapping.get(result['id'], "Unknown")
        result['id'] = original_id
    return search_results

# Define the missing function to perform search by keywords or author
def search_by_keywords_or_author(milvus_wrapper, keywords, user_query_text, limit=3):
    """
    Searches based on extracted keywords (technology/product) and author names.
    Falls back to vector search if no specific keywords are found.
    """
    author = keywords.get('author')
    entities = keywords.get('entities')

    if author:
        # First, search by author if mentioned
        author_search_result = milvus_wrapper.search_by_author(author, limit=limit)
        cleaned_results = json.loads(author_search_result)
    elif entities:
        # Search by technology keywords if mentioned or general search terms like 'entity extraction'
        keyword_search_result = milvus_wrapper.search_by_keywords(entities, limit=limit)
        cleaned_results = json.loads(keyword_search_result)
    else:
        # Fallback to vector search across all fields
        vector_search_result = milvus_wrapper.search(user_query_text, limit=limit)
        cleaned_results = milvus_wrapper.clean_result(vector_search_result)  # Convert hashed IDs to original

    # Map hashed IDs to original before returning results
    cleaned_results = map_hashed_ids_to_original(milvus_wrapper, cleaned_results)

    return cleaned_results

# Get route for health check
@app.get("/")
def read_root():
    return {"message": "FastAPI Milvus Search API is running!"}

# Post route for search functionality
@app.post("/search/")
def search(query_request: QueryRequest):
    try:
        db_path = "./milvus_database.db"
        collection_name = "asset_collection"
        csv_file_path = "./assetdata_jul-22-2024.csv"
        user_query_text = query_request.user_query_text
        limit = 3

        # Initialize MilvusWrapper
        milvus_wrapper = MilvusWrapper(db_path, collection_name)
        milvus_wrapper.delete_all_data()

        # Load and insert data
        data = milvus_wrapper.load_data(csv_file_path)
        milvus_wrapper.insert_data(data)

        # Extract relevant keywords (author and tech keywords) from the query
        keywords = extract_relevant_keywords(user_query_text)

        # Perform the search
        search_results = search_by_keywords_or_author(milvus_wrapper, keywords, user_query_text, limit=limit)

        if search_results:
            # Initialize the LLM
            llm = LLMBackendCloud()
            search_results_json = json.dumps(search_results, indent=4)
            prompt = prepare_prompt(user_query_text, search_results_json)
            response = llm.generate_response(prompt)
            # Extract all asset codes
            matches = re.findall(r'ID:\s([a-zA-Z0-9]+)', response)
            matches_dict = {f"asset_id_{i+1}": match for i, match in enumerate(matches)}
            print(matches_dict)
            # Convert to a JSON string
            # Print keys
            json_output = json.dumps(matches_dict, indent=2)
            # Convert the dictionary to a list of dictionaries
            converted_data = [{"assetId": value} for key, value in matches_dict.items()]
            return converted_data
            #return Response(content=converted_data, media_type="application/json")
        else:
            raise HTTPException(status_code=404, detail="No relevant assets found for the given query.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
