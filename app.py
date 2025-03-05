import streamlit as st
import requests

def main():
    st.set_page_config(page_title="Legal Chatbot", layout="wide")
    st.sidebar.title("Legal Chatbot Settings")
    st.sidebar.write("This chatbot helps with legal queries by retrieving relevant legal documents, summarizing them, and generating AI-enhanced responses.")
    st.title("🧑‍⚖️ Legal Chatbot")
    st.markdown("### Ask your legal question below:")
    
    user_query = st.text_input("Enter your legal query:")
    
    if st.button("Submit", use_container_width=True):
        if user_query:
            st.info("Processing your request...")
            response = requests.get(f"http://localhost:8000/query?query={user_query}").json()
            
            # Display user query
            st.markdown("### 📝 User Query")
            st.write(user_query)
            
            # Display retrieved knowledge base data
            st.markdown("### 📖 Retrieved Data from Knowledge Base")
            if response["retrieved_data"]:
                for section in response["retrieved_data"]:
                    st.markdown(f"🔹 {section}")
            else:
                st.warning("No relevant legal documents found.")

            # Display summarized text
            st.markdown("### ✨ Summarization Text")
            if response["summarized_texts"]:
                for summary in response["summarized_texts"]:
                    st.markdown(f"✅ {summary}")
            else:
                st.warning("No summarization available.")

            # Display LLM-generated response
            st.markdown("### 🤖 AI-Generated Response")
            st.text_area("Final Answer:", response["response"], height=150)

if __name__ == "__main__":
    main()
