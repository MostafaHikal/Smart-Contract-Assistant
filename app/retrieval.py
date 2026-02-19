import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

def get_contract_response(user_question):
    # 1. Load Embeddings and Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.load_local(
        "vector_store/faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # 2. Semantic Search (Retrieve top 3 chunks)
    docs = vector_db.similarity_search(user_question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 3. Initialize LLM (Groq)
    llm = ChatGroq(
        temperature=0,
        model_name=os.getenv("MODEL_NAME"), # MODEL_NAME=llama-3.1-8b-instant
        groq_api_key=os.getenv("GROG_API_KEY")
    )
    
    # 4. Construct the RAG Prompt
    full_prompt = f"""
    You are a helpful and intelligent Legal Assistant. Your mission is to assist the user by providing clear, conversational, and accurate answers based on the provided context.

    ### How to respond:
    1. Be conversational and friendly (like a smart colleague).
    2. If the answer is in the document, explain it naturally and use bullet points if it helps clarity.
    3. If the information isn't there, don't just say "Not found". Instead, say something like: "I've looked through the document, but I couldn't find any mention of [the topic]. It seems to be outside the current scope."
    4. If asked for a summary, give a warm overview that highlights the most important parts.

    ### CONTEXT:
    {context}
    
    ### USER QUESTION: 
    {user_question}
    
    ### YOUR HELPFUL RESPONSE:"""
    
    # 5. Invoke the LLM and return content
    response = llm.invoke(full_prompt)
    return response.content

if __name__ == "__main__":
    print("--- Testing RAG (Manual Way) ---")
    answer = get_contract_response("Summary of the contract?")
    print(f"Result: {answer}")