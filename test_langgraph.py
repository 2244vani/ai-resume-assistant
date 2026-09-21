from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# 1. Define the state
class State(TypedDict):
    resume: str
    result: str


# 2. First node
def check_resume(state: State):

    resume = state["resume"].lower()

    if "python" in resume:
        return {
            "result": "Python is present in the resume."
        }

    return {
        "result": "Python is not mentioned in the resume."
    }


# 3. Decision function
def decide_next(state: State):

    if "present" in state["result"]:
        return "python_found"

    return "python_missing"


# 4. Node when Python is found
def python_found(state: State):

    return {
        "result": state["result"] + " The candidate has Python experience."
    }


# 5. Node when Python is missing
def python_missing(state: State):

    return {
        "result": state["result"] + " Consider learning Python."
    }


# 6. Create the graph
graph = StateGraph(State)


# 7. Add nodes
graph.add_node("check_resume", check_resume)
graph.add_node("python_found", python_found)
graph.add_node("python_missing", python_missing)


# 8. START → Check Resume
graph.add_edge(START, "check_resume")


# 9. Conditional edge
graph.add_conditional_edges(
    "check_resume",
    decide_next,
    {
        "python_found": "python_found",
        "python_missing": "python_missing"
    }
)


# 10. Connect both paths to END
graph.add_edge("python_found", END)
graph.add_edge("python_missing", END)


# 11. Compile
app = graph.compile()


# 12. Test resume
resume = """
The candidate knows Python, SQL and Machine Learning.
"""


# 13. Run the graph
result = app.invoke({
    "resume": resume,
    "result": ""
})


# 14. Display result
print("Result:")
print(result["result"])