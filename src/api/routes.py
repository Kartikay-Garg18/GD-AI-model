from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from src.chains.transcript_analyzer import analyze_transcript_plain, analyze_transcript_structured
from src.chains.trending_topics import get_trending_gd_topics
from typing import List

router = APIRouter()

# Request model
class TranscriptRequest(BaseModel):
    transcript: str

# Response model
class TranscriptResponse(BaseModel):
    plain_text_analysis: str
    structured_analysis: dict

class GDTopic(BaseModel):
    topic: str = Field(..., description="The GD topic title")
    explanation: str = Field(..., description="Short explanation or context for the topic")

class GDTopicsResponse(BaseModel):
    topics: List[GDTopic]

@router.post("/analyze", response_model=TranscriptResponse)
def analyze_transcript(req: TranscriptRequest):
    transcript = req.transcript.strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript cannot be empty")

    plain_text = analyze_transcript_plain(transcript)
    structured = analyze_transcript_structured(transcript)
    structured_dict = structured.model_dump()  

    return TranscriptResponse(
        plain_text_analysis=plain_text,
        structured_analysis=structured_dict
    )

@router.get("/trending-gd-topics", response_model=GDTopicsResponse)
def trending_gd_topics(
    category: str = Query("general", description="Category of GD topics"),
    top_k: int = Query(5, description="Number of topics to return")
):
    result = get_trending_gd_topics(category=category, top_k=top_k)
    topics_as_gdmodels = [GDTopic(**t.model_dump()) for t in result.topics]
    return GDTopicsResponse(topics=topics_as_gdmodels)

