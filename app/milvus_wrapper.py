# Ensure you have pymilvus, numpy, pandas, and sentence-transformers installed in your environment
# pip install -U pymilvus numpy pandas sentence-transformers

from pymilvus import MilvusClient
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import json

class MilvusWrapper:
    def __init__(self, db_path, collection_name):
        """
        Initialize the MilvusWrapper with a database path and collection name.

        Parameters:
        - db_path (str): Path to the Milvus database.
        - collection_name (str): Name of the collection to operate on.
        """
        self.client = MilvusClient(db_path)  # Initialize the Milvus client
        self.collection_name = collection_name  # Set the collection name
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Load a pre-trained model for embeddings
        self.dimension = 384  # Dimension of the embeddings

    def create_collection(self):
        """
        Create a new collection in Milvus with the specified name and dimension.
        """
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=self.dimension
        )

    def delete_all_data(self):
        """
        Delete all data in the collection by dropping it and creating a new one.
        """
        try:
            if self.client.has_collection(self.collection_name):
                self.client.drop_collection(self.collection_name)
                print(f"Collection '{self.collection_name}' dropped successfully.")
                self.create_collection()  # Recreate the collection
                print(f"Collection '{self.collection_name}' recreated successfully.")
            else:
                print(f"Collection '{self.collection_name}' does not exist, creating it.")
                self.create_collection()
        except Exception as e:
            print(f"Failed to delete or recreate collection '{self.collection_name}':", e)

    def load_data(self, csv_file_path):
        """
        Load data from a CSV file and prepare it for insertion into Milvus.

        Parameters:
        - csv_file_path (str): Path to the CSV file containing the data.

        Returns:
        - data (list): List of dictionaries containing the prepared data for insertion.
        """
        df = pd.read_csv(csv_file_path)  # Read the CSV file into a DataFrame
        df['id'] = df['id'].apply(lambda x: hash(x) % (10 ** 8))  # Convert 'id' to a hashed integer
        df['vector'] = df['description'].apply(
            lambda x: self.model.encode(x) if pd.notnull(x) else np.zeros(self.dimension)
        )  # Generate vectors for the 'description' field

        # Prepare the data for insertion into Milvus
        data = [
            {
                "id": row['id'],
                "vector": row['vector'].tolist(),
                "title": row['title'],
                "author": row['author'],
                "description": row['description'],
                "type": row['type']
            }
            for index, row in df.iterrows()
        ]

        return data

    def insert_data(self, data):
        """
        Insert data into the Milvus collection.

        Parameters:
        - data (list): List of dictionaries containing the data to insert.
        """
        res = self.client.insert(
            collection_name=self.collection_name,
            data=data
        )
        #print("Data Insertion Result:", res)
        print("Data Insertion Completed")

    def search(self, query_text, limit=3):
        """
        Perform a vector similarity search in the Milvus collection.

        Parameters:
        - query_text (str): Text query to search for.
        - limit (int): Number of search results to return.

        Returns:
        - res (list): Raw search results from Milvus.
        """
        query_vector = self.model.encode(query_text).tolist()  # Generate vector for the query text
        try:
            res = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector],  # Ensure data is a list of lists
                vector_field="vector",  # Specify vector field
                limit=limit,
                output_fields=["title", "author", "description", "type"],
            )
            #print("Search Result:", res)
            print("Search Completed successfully")
            return res
        except Exception as e:
            print("Failed to perform search:", e)

    # def hybrid_search(self, query_text, filter_criteria=None, limit=3):
    #     """
    #     Perform a hybrid search that combines vector similarity and scalar filtering.

    #     Parameters:
    #     - query_text (str): Text query to search for.
    #     - filter_criteria (dict): Dictionary of field-value pairs for filtering results.
    #     - limit (int): Number of search results to return.

    #     Returns:
    #     - res (list): Raw search results from Milvus.
    #     """
    #     query_vector = self.model.encode(query_text).tolist()  # Generate vector for the query text

    #     # Ensure filter criteria is formatted correctly
    #     formatted_filter_criteria = None
    #     if filter_criteria:
    #         try:
    #             formatted_filter_criteria = " AND ".join(
    #                 [f"{key} == '{value}'" for key, value in filter_criteria.items()]
    #             )
    #         except Exception as e:
    #             print("Failed to format filter criteria:", e)
    #             return []

    #     # Build search parameters
    #     search_params = {
    #         "limit": limit,  # Limit of results to return
    #         "output_fields": ["title", "author", "description", "type"],  # Fields to return
    #     }

    #     # If filter criteria are provided, add them to the search parameters
    #     if formatted_filter_criteria:
    #         search_params["filter"] = formatted_filter_criteria

    #     try:
    #         # Perform the hybrid search with collection name and data as required arguments
    #         res = self.client.search(
    #             collection_name=self.collection_name,
    #             data=[query_vector],  # Ensure data is a list of lists
    #             vector_field="vector",  # Specify vector field
    #             **search_params
    #         )
    #         print("Hybrid Search Result:", res)
    #         return res
    #     except Exception as e:
    #         print("Failed to perform hybrid search:", e)
    #         return []

    @staticmethod
    def clean_result(result):
        """
        Clean and format search results for better readability and JSON output.

        Parameters:
        - result (list): The raw result from a search or query operation.

        Returns:
        - cleaned_results (list): List of dictionaries with cleaned data.
        """
        cleaned_results = []

        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict):
                    entity = item.get('entity', {})
                    cleaned_entry = {
                        'id': item.get('id'),
                        'distance': item.get('distance'),
                        'title': entity.get('title'),
                        'author': entity.get('author'),
                        'description': entity.get('description'),
                        'type': entity.get('type')
                    }
                    cleaned_results.append(cleaned_entry)
                elif isinstance(item, list):
                    for sub_item in item:
                        if isinstance(sub_item, dict):
                            entity = sub_item.get('entity', {})
                            cleaned_entry = {
                                'id': sub_item.get('id'),
                                'distance': sub_item.get('distance'),
                                'title': entity.get('title'),
                                'author': entity.get('author'),
                                'description': entity.get('description'),
                                'type': entity.get('type')
                            }
                            cleaned_results.append(cleaned_entry)
        else:
            print("The input result is not in the expected format.")
        return cleaned_results

    def close(self):
        """
        Close the connection to the Milvus client.
        """
        self.client.close()
        print("Milvus connection closed.")
