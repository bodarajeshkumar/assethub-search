from fastapi import FastAPI
from pydantic import BaseModel
from milvus_wrapper import MilvusWrapper
from fastapi.middleware.cors import CORSMiddleware
import warnings

warnings.filterwarnings("ignore")

app = FastAPI()
app.add_middleware(
    ["*"],
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define paths and collection details
db_path = "./milvus_database.db"
collection_name = "asset_collection"
csv_file_path = "./assetdata_jul-22-2024.csv"

# Initialize MilvusWrapper
milvus_wrapper = MilvusWrapper(db_path, collection_name)

# Pydantic model to handle the incoming query request
class QueryRequest(BaseModel):
    query_text: str

# On startup, load and insert data from the CSV file into Milvus
@app.on_event("startup")
async def startup_event():
    milvus_wrapper.delete_all_data()
    data = milvus_wrapper.load_data(csv_file_path)
    milvus_wrapper.insert_data(data)

# On shutdown, close the Milvus connection
@app.on_event("shutdown")
async def shutdown_event():
    milvus_wrapper.close()

# POST endpoint to perform vector similarity search
@app.post("/search")
async def search(query: QueryRequest):
    vector_search_result = milvus_wrapper.search(query.query_text)
    cleaned_vector_search_result = milvus_wrapper.clean_result(vector_search_result)
    return {"result": cleaned_vector_search_result}

# Root endpoint for health check or API info
@app.get("/")
async def read_root():
    return {"message": "Milvus Search API is up and running!"}
