from langchain_google_genai import ChatGoogleGenerativeAI
from src.config.settings import settings

# Initialize LLM (Gemini)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=settings.GEMINI_API_KEY
)
