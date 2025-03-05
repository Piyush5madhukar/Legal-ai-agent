
##  Architecture of the Multi-Agent System  

Your legal chatbot follows a multi-agent system where different components handle specific tasks. The architecture consists of three main agents:  

a. Query Agent  
- Loads the FAISS index and legal documents.  
- Encodes the user query using SentenceTransformer and searches for the most relevant legal documents.  
- Returns the top `k` retrieved legal texts.  

b. Summarization Agent  
- Uses the Facebook BART-Large-CNN model to summarize the retrieved legal documents.  
- Runs the summarization in parallel using ThreadPoolExecutor to speed up processing.  

c. LLM Agent 
- Uses Google Gemini 2.0 Flash to generate a final response based on the summarized text.  
- Ensures the LLM response is concise by truncating input to 1500 characters.  

These agents work together to process a legal query, retrieve relevant documents, summarize them, and generate an AI-powered response.  



2. Code Implementation for Each Agent 

### **Query Agent (`query_agent.py`)  
- Loads FAISS index and document texts.  
- Retrieves top-k relevant legal sections.  

```python
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from typing import List

INDEX_PATH = "faiss_index.idx"
DOCS_PATH = "doc_texts.npy"

if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
    index = faiss.read_index(INDEX_PATH)
    doc_texts = np.load(DOCS_PATH, allow_pickle=True)
else:
    raise FileNotFoundError("FAISS index or document data not found. Run indexing first.")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

class UserQuery(BaseModel):
    query: str

class RetrievedSection(BaseModel):
    text: str

def query_legal_documents(query: UserQuery, top_k=3) -> List[RetrievedSection]:
    query_embedding = embedding_model.encode([query.query], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(query_embedding, top_k)
    return [RetrievedSection(text=doc_texts[i]) for i in indices[0] if i < len(doc_texts)]
```

---

Summarization Agent (`summarization_agent.py`)  
- Uses Facebook’s **BART** model to summarize retrieved texts.  
- Implements parallel summarization using `ThreadPoolExecutor`.  

```python
from transformers import pipeline
from typing import List
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from query_agent import RetrievedSection

summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

class SummarizedResponse(BaseModel):
    summary: str

def summarize_text(text_sections: List[RetrievedSection]) -> List[SummarizedResponse]:
    with ThreadPoolExecutor(max_workers=4) as executor:
        summaries = list(executor.map(lambda section: summarizer(section.text, max_length=100, min_length=30, do_sample=False)[0]["summary_text"], text_sections))
    return [SummarizedResponse(summary=s) for s in summaries]
```

---

LLM Agent (`llm_agent.py`)  
- Uses Google Gemini 2.0 Flash to generate a response based on summarized text.  
- Truncates input to 1500 characters for efficiency.  

```python
import google.generativeai as genai
from dotenv import load_dotenv
import os
from pydantic import BaseModel

load_dotenv()
api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)

gemini_model = genai.GenerativeModel(model_name="gemini-2.0-flash")

class FinalLLMResponse(BaseModel):
    response: str

def generate_llm_response(summary_text: str) -> FinalLLMResponse:
    truncated_text = summary_text[:1500]
    response = gemini_model.generate_content(f"Provide a brief response. Do not use any special formatting. Here is the input:\n\n{truncated_text}")
    return FinalLLMResponse(response=response.text)
```

---

FastAPI Backend (`main.py`)  
- Calls the Query, Summarization, and LLM agents sequentially.  

```python
from fastapi import FastAPI
from query_agent import query_legal_documents
from summarization_agent import summarize_text
from llm_agent import generate_llm_response
from query_agent import UserQuery

app = FastAPI()

@app.post("/query")
def legal_chatbot(query: UserQuery):
    retrieved_texts = query_legal_documents(query)
    
    if not retrieved_texts:
        return {"response": "No relevant legal information found.", "retrieved_data": [], "summarized_texts": []}

    summarized_texts = summarize_text(retrieved_texts)
    combined_summary = "\n".join([s.summary for s in summarized_texts])
    llm_response = generate_llm_response(combined_summary)

    return {
        "retrieved_data": [s.text for s in retrieved_texts],
        "summarized_texts": [s.summary for s in summarized_texts],
        "response": llm_response.response
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

Streamlit Frontend (`app.py`)  
- Provides a simple UI for users to enter legal queries.  
- Calls FastAPI to retrieve and display results.  

```python
import streamlit as st
import requests

API_URL = "http://localhost:8000/query"

def main():
    st.set_page_config(page_title="Legal Chatbot", layout="wide")
    st.sidebar.title("Legal Chatbot Settings")
    st.sidebar.write("This chatbot helps with legal queries by retrieving relevant legal documents, summarizing them, and generating AI-enhanced responses.")
    
    st.title("🧑‍⚖️ Legal Chatbot")
    user_query = st.text_input("Enter your legal query:")
    
    if st.button("Submit"):
        if user_query:
            st.info("Processing your request...")
            try:
                response = requests.post(API_URL, json={"query": user_query}).json()

                st.markdown("### User Query")
                st.write(user_query)

                st.markdown("###  Retrieved Data")
                for section in response["retrieved_data"]:
                    st.markdown(f" {section}")

                st.markdown("###  Summarized Text")
                for summary in response["summarized_texts"]:
                    st.markdown(f"{summary}")

                st.markdown("###  AI-Generated Response")
                st.text_area("Final Answer:", response["response"], height=150)

            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {str(e)}")

if __name__ == "__main__":
    main()
```

---

3. Demo Showcasing Chatbot Responses 

1. User enters a query: `"What are the legal rights of a tenant?"`  
2. FAISS retrieves relevant legal texts. 
3. Summarization agent condenses the legal texts into a brief summary.
4. LLM generates a final response. 

Example Response:  
Retrieved Data:  
- "A tenant has rights including habitability, privacy, and protection from wrongful eviction..."  

Summarized Text:  
- "Tenants have the right to a livable environment, legal protection from eviction, and privacy."  
 AI-Generated Response:  
- "Tenants are entitled to safe living conditions, privacy, and protection under housing laws. Let me know if you need specifics on eviction rights or rent control."  

---

4. Challenges Faced & Possible Improvements 

Challenges  
- Handling long documents: Some legal texts exceed the LLM input limit, requiring truncation.  
- Processing time: Summarization can be slow for multiple documents.  
- Query accuracy: FAISS retrieval depends on embedding quality.  

Possible Improvements  
- Chunking large document before retrieval to improve search relevance.  
- Caching responses to reduce redundant computations.  
- Fine-tuning embeddings using domain-specific legal data.  

---
 Conclusion  
This multi-agent system efficiently retrieves, summarizes, and responds to legal queries. With further optimizations, it can provide faster and more accurate legal assistance. 
