from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# 1. Load the resume
loader = PyPDFLoader("uploaded_resume.pdf")

documents = loader.load()

print("Resume loaded successfully!")


# 2. Split the resume
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# 3. Create embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# 4. Create vector store
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="final_resume_workflow"
)

print("Resume stored successfully!")


# 5. Ask a question
question = "What programming languages does the candidate know?"


# 6. Search the resume
results = vector_store.similarity_search(
    question,
    k=3
)


# 7. Display retrieved information
print("\nQuestion:")
print(question)

print("\nRelevant Resume Information:")

for i, result in enumerate(results):

    print(f"\n--- Result {i + 1} ---")
    print(result.page_content)