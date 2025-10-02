from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field  
from src.models.gemini_model import llm


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant that suggests trending Group Discussion (GD) topics."),
    ("human", "Suggest {top_k} trending GD topics in the category '{category}'. Return as a list with explanations.")
])
chain = prompt | llm


class GDTopic(BaseModel):
    topic: str = Field(..., description="The GD topic title")
    explanation: str = Field(..., description="Short explanation or context for the topic")

class TrendingGDTopics(BaseModel):
    topics: list[GDTopic] = Field(..., description="List of trending GD topics with explanations")

structured_llm = llm.with_structured_output(TrendingGDTopics, method="json_mode")


def get_trending_gd_topics(category: str = "general", top_k: int = 5) -> TrendingGDTopics:
    """
    Returns structured trending GD topics using LLM.
    """
    formatted_prompt = prompt.format(category=category, top_k=top_k)

    result = structured_llm.invoke(formatted_prompt)

    if isinstance(result, dict):
        return TrendingGDTopics(**result)
    elif isinstance(result, TrendingGDTopics):
        return result
    else:
        raise TypeError(f"Unexpected output type: {type(result)}")
