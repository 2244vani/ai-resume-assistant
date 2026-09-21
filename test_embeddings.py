from langchain_ollama import OllamaEmbeddings


embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


text = "Python developer with machine learning experience"

vector = embeddings.embed_query(text)

print("Text:")
print(text)

print("\nEmbedding vector:")
print(vector)

print("\nNumber of values:")
print(len(vector))