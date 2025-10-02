from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field  
from src.models.gemini_model import llm


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant that analyzes meeting transcripts."),
    ("human", "Analyze the following transcript: {transcript}")
])
chain = prompt | llm


class TranscriptAnalysis(BaseModel):
    key_points: str = Field(..., description="Important points from the transcript")
    sentiment: str = Field(..., description="Overall sentiment of the transcript")
    recommendations: str = Field(..., description="Actionable recommendations")

structured_llm = llm.with_structured_output(TranscriptAnalysis, method="json_mode")


def analyze_transcript_plain(transcript: str) -> str:
    result = chain.invoke({"transcript": transcript})
    if isinstance(result.content, str):
        return result.content
    else:
        return str(result.content)


def analyze_transcript_structured(transcript: str) -> TranscriptAnalysis:
    result = structured_llm.invoke(transcript)
    if isinstance(result, dict):
        return TranscriptAnalysis(**result)
    elif isinstance(result, TranscriptAnalysis):
        return result
    else:
        raise TypeError(f"Unexpected output type: {type(result)}")
