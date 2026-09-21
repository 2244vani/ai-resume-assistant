from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings


# 1. Load the resume PDF
loader = PyPDFLoader("uploaded_resume.pdf")

documents = loader.load()

print("Number of pages:", len(documents))


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


# 4. Convert each chunk into an embedding
texts = [chunk.page_content for chunk in chunks]

vectors = embeddings.embed_documents(texts)


# 5. Show the result
print("Number of vectors:", len(vectors))

print("Number of values in first vector:", len(vectors[0]))