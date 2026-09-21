from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from typing import TypedDict


# 1. Define the state
class State(TypedDict):
    question: str
    answer: str


# 2. Create the calculator tool
@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""

    try:
        return str(eval(expression))
    except Exception:
        return "Could not calculate the expression."


# 3. Create Ollama model
model = ChatOllama(
    model="llama3.2"
)


# 4. Create model with the calculator tool
model_with_tools = model.bind_tools(
    [calculator]
)


# --------------------------------
# Node 1: Ask the model
# --------------------------------

def ask_model(state: State):

    response = model_with_tools.invoke(
        state["question"]
    )

    # Check whether the model selected a tool
    if response.tool_calls:

        tool_call = response.tool_calls[0]

        result = calculator.invoke(
            tool_call["args"]
        )

        return {
            "answer": result
        }

    # No tool needed
    return {
        "answer": response.content
    }


# 5. Create graph
graph = StateGraph(State)


# 6. Add node
graph.add_node(
    "ask_model",
    ask_model
)


# 7. Connect nodes
graph.add_edge(
    START,
    "ask_model"
)

graph.add_edge(
    "ask_model",
    END
)


# 8. Compile
app = graph.compile()


# 9. Ask question
question = "What is 25 + 75?"


# 10. Run graph
result = app.invoke({
    "question": question,
    "answer": ""
})


# 11. Display result
print("Question:")
print(result["question"])

print("\nFinal Answer:")
print(result["answer"])