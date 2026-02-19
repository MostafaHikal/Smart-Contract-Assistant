import os
import time
import random
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Import your actual RAG function
from app.retrieval import get_contract_response

load_dotenv()

# 1. Configuration & Models Setup
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db_path = "vector_store/faiss_index"
api_key = os.getenv("GROG_API_KEY")

# Teacher & Judge Model (70B)
llm_70b = ChatGroq(
    temperature=0.2,
    model_name="llama-3.3-70b-versatile",
    api_key=api_key
)

# --- PHASE 1: DATA SAMPLING ---
def get_random_chunks_from_db(num_chunks=7):
    vector_db = FAISS.load_local(
        vector_db_path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    all_chunks = list(vector_db.docstore._dict.values())
    actual_num = min(num_chunks, len(all_chunks))
    random_samples = random.sample(all_chunks, actual_num)
    
    contexts = [doc.page_content for doc in random_samples]
    print(f"Successfully sampled {len(contexts)} chunks.")
    return contexts

# --- PHASE 2: QUESTION GENERATION ---
def generate_synthetic_qa(contexts):
    qa_dataset = []
    template = """
    You are a Senior Legal Examiner. Your task is to generate ONE specific question and its concise answer based ONLY on the provided contract snippet.
    
    Rules:
    - The question must be factual and directly answerable from the text.
    - The answer must be a short, authoritative extract from the text.
    - Output format must be: Question: [Insert Question] | Answer: [Insert Answer]
    
    Contract Snippet:
    {context}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm_70b
    
    print(f"Generating {len(contexts)} QA pairs...")
    for i, ctx in enumerate(contexts):
        try:
            response = chain.invoke({"context": ctx})
            raw_output = response.content
            
            if "|" in raw_output:
                parts = raw_output.split("|")
                question = parts[0].replace("Question:", "").strip()
                answer = parts[1].replace("Answer:", "").strip()
                
                qa_dataset.append({
                    "question": question,
                    "ground_truth": answer,
                    "context": ctx
                })
                print(f"Generated QA Pair {i+1} successfully.")
            
            # Rate limiting for Groq API
            time.sleep(1.5)
            
        except Exception as e:
            print(f"Error generating QA pair {i+1}: {e}")
            time.sleep(2.0)
            
    return qa_dataset

# --- PHASE 3: RAG EXECUTION ---
def run_rag_on_questions(qa_dataset):
    print(f"Testing RAG system on {len(qa_dataset)} questions...")
    for i, item in enumerate(qa_dataset):
        try:
            # Using the actual logic from retrieval.py
            rag_answer = get_contract_response(item['question'])
            item['rag_answer'] = rag_answer
            print(f"RAG answered question {i+1}/{len(qa_dataset)}")
        except Exception as e:
            item['rag_answer'] = f"Error during retrieval: {str(e)}"
    return qa_dataset

# --- PHASE 4: EVALUATION (THE JUDGE) ---
def evaluate_answers(qa_dataset):
    print(f"Judging {len(qa_dataset)} responses...")
    
    eval_prompt = ChatPromptTemplate.from_template("""INSTRUCTION: 
    You are an objective evaluator comparing two answers to the same question.

    Evaluate the following Question-Answer pair:

    {qa_trio}

    Evaluation Criteria:
    - Answer 1 is the baseline (limited context, shorter)
    - Answer 2 is from a RAG system (full context, may be longer)
    - Answer 2 should be scored [2] if it:
      * Provides accurate information
      * Is more detailed and comprehensive than Answer 1
      * Uses the broader context appropriately
      * Does not contradict Answer 1's core facts
      
    - Answer 2 should be scored [1] ONLY if it:
      * Contains factually incorrect information
      * Completely fails to answer the question
      * Contradicts the baseline answer on key facts

    Output EXACTLY in this format:
    [Score] Justification

    EVALUATION: 
    """)

    judge_chain = eval_prompt | llm_70b
    
    for i, item in enumerate(qa_dataset):
        try:
            qa_trio_text = (
                f"Question: {item['question']}\n\n"
                f"Answer 1 (Baseline): {item['ground_truth']}\n\n"
                f"Answer 2 (RAG System): {item['rag_answer']}"
            )
            
            response = judge_chain.invoke({'qa_trio': qa_trio_text})
            eval_output = response.content
            
            if "]" in eval_output:
                score_part = eval_output.split("]")[0].replace("[", "").strip()
                reason_part = eval_output.split("]")[1].strip()
                item['score'] = score_part
                item['reason'] = reason_part
            else:
                item['score'] = "N/A"
                item['reason'] = eval_output
            
            # Rate limiting for the Judge
            time.sleep(2.0)
            print(f"Judged pair {i+1} successfully.")
            
        except Exception as e:
            print(f"Error judging pair {i+1}: {e}")
            item['score'] = "1"
            item['reason'] = f"Evaluation Error: {str(e)}"
            
    return qa_dataset

# --- FINAL ORCHESTRATOR ---
def run_full_evaluation_process():
    sampled_contexts = get_random_chunks_from_db(7)
    qa_dataset = generate_synthetic_qa(sampled_contexts)
    
    if not qa_dataset:
        return [], 0, 0, 0
        
    qa_dataset_with_rag = run_rag_on_questions(qa_dataset)
    final_results = evaluate_answers(qa_dataset_with_rag)
    
    total_questions = len(final_results)
    passed_questions = sum(1 for item in final_results if item.get('score') == '2')
    
    accuracy_percentage = (passed_questions / total_questions * 100) if total_questions > 0 else 0
    
    return final_results, accuracy_percentage, passed_questions, total_questions

if __name__ == "__main__":
    results, acc, passed, total = run_full_evaluation_process()
    
    print("\n" + "="*50)
    print("         RAG PERFORMANCE SUMMARY")
    print("="*50)
    print(f"TOTAL SCORE: {passed}/{total} ({acc:.1f}%)") 
    print("="*50)
    
    for res in results:
        status = "✅ PASS" if res.get('score') == '2' else "❌ FAIL"
        print(f"\n[{status}] Q: {res.get('question')}")
        print(f"Reason: {res.get('reason')}")