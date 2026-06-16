import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collections = client.list_collections()

print("=" * 60)
print("ChromaDB Collections")
print("=" * 60)

for col in collections:
    print(col.name)

print("=" * 60)

for col in collections:
    collection = client.get_collection(col.name)

    print(f"{col.name} : {collection.count()} chunks")