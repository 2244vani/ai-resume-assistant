from langgraph.graph import StateGraph, START, END
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from typing import TypedDict


# --------------------------------
# 1. Define the state
# --------------------------------

class State(TypedDict):
    question: str
    context: str
    answer: str


# --------------------------------
# 2. Load the resume
# --------------------------------

loader = PyPDFLoader("uploaded_resume.pdf")

documents = loader.load()

print("Resume loaded successfully!")


# --------------------------------
# 3. Split the resume
# --------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# --------------------------------
# 4. Create embeddings
# --------------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# --------------------------------
# 5. Create vector store
# --------------------------------

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="langgraph_conditional_resume"
)

print("Resume stored in vector database!")


# --------------------------------
# 6. Create Llama model
# --------------------------------

model = ChatOllama(
    model="llama3.2"
)


# --------------------------------
# Node 1: Retrieve resume
# --------------------------------

def retrieve_resume(state: State):

    results = vector_store.similarity_search(
        state["question"],
        k=3
    )

    context = "\n\n".join(
        result.page_content
        for result in results
    )

    return {
        "context": context
    }


# --------------------------------
# Node 2: Check information
# --------------------------------

def check_information(state: State):

    if not state["context"].strip():

        return {
            "answer": "No relevant information was found."
        }

    return {
        "answer": "Information found."
    }


# --------------------------------
# Decision function
# --------------------------------

def decide_next(state: State):

    if state["answer"] == "Information found.":

        return "generate_answer"

    return "no_information"


# --------------------------------
# Node 3: Generate answer
# --------------------------------

def generate_answer(state: State):

    prompt = f"""
    Answer the question using only the resume information.

    Resume information:
    {state["context"]}

    Question:
    {state["question"]}

    Give a simple and clear answer.
    """

    response = model.invoke(prompt)

    return {
        "answer": response.content
    }


# --------------------------------
# Node 4: No information
# --------------------------------

def no_information(state: State):

    return {
        "answer": "Sorry, I could not find relevant information in the resume."
    }


# --------------------------------
# 7. Create graph
# --------------------------------

graph = StateGraph(State)


# --------------------------------
# 8. Add nodes
# --------------------------------

graph.add_node(
    "retrieve_resume",
    retrieve_resume
)

graph.add_node(
    "check_information",
    check_information
)

graph.add_node(
    "generate_answer",
    generate_answer
)

graph.add_node(
    "no_information",
    no_information
)


# --------------------------------
# 9. Connect nodes
# --------------------------------

graph.add_edge(
    START,
    "retrieve_resume"
)

graph.add_edge(
    "retrieve_resume",
    "check_information"
)


# --------------------------------
# 10. Conditional edge
# --------------------------------

graph.add_conditional_edges(
    "check_information",
    decide_next,
    {
        "generate_answer": "generate_answer",
        "no_information": "no_information"
    }
)


# --------------------------------
# 11. Connect to END
# --------------------------------

graph.add_edge(
    "generate_answer",
    END
)

graph.add_edge(
    "no_information",
    END
)


# --------------------------------
# 12. Compile graph
# --------------------------------

app = graph.compile()


# --------------------------------
# 13. Ask a question
# --------------------------------

question = "What programming languages does the candidate know?"


# --------------------------------
# 14. Run workflow
# --------------------------------

result = app.invoke({
    "question": question,
    "context": "",
    "answer": ""
})


# --------------------------------
# 15. Display answer
# --------------------------------

print("\nQuestion:")
print(result["question"])

print("\nFinal Answer:")
print(result["answer"])