from transformers import pipeline
from typing import List
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from query_agent import RetrievedSection

# Load summarization model (Optimized for CPU)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

class SummarizedResponse(BaseModel):
    summary: str

def summarize_text(text_sections: List[RetrievedSection]) -> List[SummarizedResponse]:
    """Summarize multiple text sections in parallel."""
    with ThreadPoolExecutor(max_workers=4) as executor:
        summaries = list(executor.map(lambda section: summarizer(section.text, max_length=100, min_length=30, do_sample=False)[0]["summary_text"], text_sections))
    return [SummarizedResponse(summary=s) for s in summaries]