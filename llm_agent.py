import google.generativeai as genai
from dotenv import load_dotenv
import os
from query_agent import BaseModel

# Load API key once
load_dotenv()
api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)

gemini_model = genai.GenerativeModel(model_name="gemini-2.0-flash")

class FinalLLMResponse(BaseModel):
    response: str

def generate_llm_response(summary_text: str) -> FinalLLMResponse:
    """Generate refined AI response using Gemini."""
    truncated_text = summary_text[:1500]  # Truncate input for faster LLM processing
    response = gemini_model.generate_content(f"Provide a **brief** response. Do not use any special formatting like **. Here is the input:\n\n{truncated_text}")
    return FinalLLMResponse(response=response.text)