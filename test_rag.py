from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
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
    collection_name="resume_rag"
)

print("Resume stored in vector database!")


# 5. User's question
question = "What programming languages does the candidate know?"

print("\nQuestion:")
print(question)


# 6. Retrieve relevant resume chunks
results = vector_store.similarity_search(
    question,
    k=3
)

print("\nRelevant information retrieved!")


# 7. Combine the retrieved chunks
context = "\n\n".join(
    result.page_content for result in results
)


# 8. Create the Ollama model
model = ChatOllama(
    model="llama3.2"
)


# 9. Give the retrieved information to Ollama
prompt = f"""
Answer the question using only the resume information provided below.

Resume information:
{context}

Question:
{question}

Give a simple and clear answer.
"""


# 10. Generate the final answer
response = model.invoke(prompt)


print("\nFinal Answer:")
print(response.content)