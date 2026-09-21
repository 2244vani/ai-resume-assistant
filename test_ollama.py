from langchain_ollama import ChatOllama

print("Starting Ollama test...")

model = ChatOllama(
    model="llama3.2"
)

print("Sending question...")

response = model.invoke(
    "Explain what a resume is in one simple sentence."
)

print("Response received:")
print(response.content)