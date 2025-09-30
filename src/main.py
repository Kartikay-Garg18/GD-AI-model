from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(title="LangChain Transcript Analyzer")

# Include routes
app.include_router(router, prefix="/api")

# Run with: uvicorn src.main:app --reload
