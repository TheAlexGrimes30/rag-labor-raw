from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
client.delete_collection("labor_dense_collection_2")
print(client.get_collections())