from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load .env file
load_dotenv()

# Create Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)
# Send a question
response = model.invoke(
    "Explain what a resume is in one simple sentence."
)

# Print the answer
print(response.content)