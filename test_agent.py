from langchain_ollama import ChatOllama
from langchain_core.tools import tool


# 1. Create calculator tool
@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception:
        return "Could not calculate the expression."


# 2. Create Ollama model
model = ChatOllama(
    model="llama3.2"
)


# 3. Give the model access to the calculator
model_with_tools = model.bind_tools(
    [calculator]
)


# 4. Ask the question
question = "What is 25 + 75?"

response = model_with_tools.invoke(question)


# 5. Check whether the model requested a tool
if response.tool_calls:

    tool_call = response.tool_calls[0]

    print("Tool selected:")
    print(tool_call["name"])

    print("\nTool input:")
    print(tool_call["args"])


    # 6. Execute the calculator
    result = calculator.invoke(
        tool_call["args"]
    )


    print("\nTool result:")
    print(result)

else:

    print("The model did not select a tool.")