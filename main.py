from fastapi import FastAPI
from query_agent import query_legal_documents
from summarization_agent import summarize_text
from llm_agent import generate_llm_response
from query_agent import UserQuery

app = FastAPI()

@app.get("/query")
def legal_chatbot(query: str):
    retrieved_texts = query_legal_documents(UserQuery(query=query))
    
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
