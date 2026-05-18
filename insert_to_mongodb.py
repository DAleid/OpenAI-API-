import os
import json
from pymongo import MongoClient

# Config
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "your_database"
COLLECTION_NAME = "your_collection"

# Resolve the json/ folder relative to this script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_FOLDER = os.path.join(BASE_DIR, "json")


def insert_json_file(filename: str):
    filepath = os.path.join(JSON_FOLDER, filename)

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]

    if isinstance(data, list):
        result = collection.insert_many(data)
        print(f"Inserted {len(result.inserted_ids)} documents from '{filename}'")
    else:
        result = collection.insert_one(data)
        print(f"Inserted 1 document from '{filename}' with id: {result.inserted_id}")

    client.close()


if __name__ == "__main__":
    json_files = [f for f in os.listdir(JSON_FOLDER) if f.endswith(".json")]

    if not json_files:
        print(f"No JSON files found in: {JSON_FOLDER}")
    else:
        for file in json_files:
            insert_json_file(file)
