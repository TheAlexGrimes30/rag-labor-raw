from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
#client.delete_collection("labor_law")
print(client.get_collections())