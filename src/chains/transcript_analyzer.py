from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field  # ✅ Use Pydantic v2
from src.models.gemini_model import llm

# ----------------------------
# Prompt Template
# ----------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant that analyzes meeting transcripts."),
    ("human", "Analyze the following transcript: {transcript}")
])
chain = prompt | llm

# ----------------------------
# Structured Output Model
# ----------------------------
class TranscriptAnalysis(BaseModel):
    key_points: str = Field(..., description="Important points from the transcript")
    sentiment: str = Field(..., description="Overall sentiment of the transcript")
    recommendations: str = Field(..., description="Actionable recommendations")

structured_llm = llm.with_structured_output(TranscriptAnalysis, method="json_mode")

# ----------------------------
# Analyzer Functions
# ----------------------------
def analyze_transcript_plain(transcript: str) -> str:
    result = chain.invoke({"transcript": transcript})
    return result.content

def analyze_transcript_structured(transcript: str) -> TranscriptAnalysis:
    return structured_llm.invoke(transcript)
