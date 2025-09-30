from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field  # ✅ Use Pydantic v2
from src.models.gemini_model import llm

# ----------------------------
# Prompt Template
# ----------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant that suggests trending Group Discussion (GD) topics."),
    ("human", "Suggest {top_k} trending GD topics in the category '{category}'. Return as a list with explanations.")
])
chain = prompt | llm

# ----------------------------
# Structured Output Model
# ----------------------------
class GDTopic(BaseModel):
    topic: str = Field(..., description="The GD topic title")
    explanation: str = Field(..., description="Short explanation or context for the topic")

class TrendingGDTopics(BaseModel):
    topics: list[GDTopic] = Field(..., description="List of trending GD topics with explanations")

# ✅ Use Pydantic v2 models with LangChain structured output
structured_llm = llm.with_structured_output(TrendingGDTopics, method="json_mode")

# ----------------------------
# Analyzer Functions
# ----------------------------
def get_trending_gd_topics(category: str = "general", top_k: int = 5) -> TrendingGDTopics:
    """
    Returns structured trending GD topics using LLM.
    """
    # Fill the prompt placeholders
    formatted_prompt = prompt.format(category=category, top_k=top_k)

    # Invoke structured LLM
    return structured_llm.invoke(formatted_prompt)
