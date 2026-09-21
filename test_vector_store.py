from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# 1. Load the resume
loader = PyPDFLoader("uploaded_resume.pdf")
documents = loader.load()

print("Resume loaded successfully!")


# 2. Split the resume into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# 3. Create the embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# 4. Create the vector store
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="resume_collection"
)

print("Resume stored in vector database successfully!")


# 5. Ask a question
question = "What programming languages does the candidate know?"

print("\nQuestion:")
print(question)


# 6. Search the vector store
results = vector_store.similarity_search(
    question,
    k=3
)


# 7. Display the matching chunks
print("\nRelevant resume sections:")

for i, result in enumerate(results):
    print("\n--- Result", i + 1, "---")
    print(result.page_content)