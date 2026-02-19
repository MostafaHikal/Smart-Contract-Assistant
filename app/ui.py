import gradio as gr
import requests
import os
from app.ingestion import ingest_document
from app.evaluator import run_full_evaluation_process 

# Configuration
API_URL = "http://127.0.0.1:8000/ask"
INDEX_PATH = "vector_store/faiss_index/index.faiss"

def handle_file_upload(file):
    """Handles file validation and ingestion with clean user messages."""
    if file is None:
        return "⚠️ Please select a file first."
    
    ext = os.path.splitext(file.name)[-1].lower()
    if ext not in [".pdf", ".docx"]:
        return "❌ Error: Only PDF and DOCX files are supported."
    
    try:
        # Call ingestion and check the internal status
        status = ingest_document(file.name)
        
        if status == "UPLOADED":
            return "✅ Document ready! You can now start chatting with the AI."
        else:
            return f"❌ System Error: {status}"
            
    except Exception as e:
        return f"❌ Ingestion Failed: {str(e)}"

def ask_bot(question):
    """Chat logic with safety checks for missing database."""
    # Safety Check: Prevent 500 Error if no file is uploaded
    if not os.path.exists(INDEX_PATH):
        return "⚠️ No document found. Please upload a contract on the left to begin."
        
    if not question.strip():
        return "Please enter a question."

    try:
        response = requests.post(API_URL, json={"question": question}, timeout=30)
        if response.status_code == 200:
            return response.json().get("answer", "No answer found.")
        else:
            return "⚠️ Backend Error: The server encountered an issue processing this document."
    except Exception:
        return "⚠️ Connection Error: Please ensure the FastAPI server is running."

def run_eval_ui():
    """Audit logic with safety checks."""
    # Safety Check: Prevent Audit crash
    if not os.path.exists(INDEX_PATH):
        return "## ⚠️ Audit Failed\nPlease upload and process a document before running the performance audit.", []

    try:
        results, acc, passed, total = run_full_evaluation_process()
        
        table_data = []
        for res in results:
            score = res.get('score', '1')
            icon = "✅" if score == '2' else "❌"
            table_data.append([
                icon, 
                res.get('question', 'N/A'), 
                score, 
                res.get('reason', 'N/A')
            ])
        
        summary_text = f"## 📊 Final Audit Score: {passed}/{total} ({acc:.1f}%)"
        return summary_text, table_data
    except Exception as e:
        return f"## ⚠️ Evaluation Failed: {str(e)}", []

# --- UI Construction ---
with gr.Blocks(theme=gr.themes.Soft(), title="Smart Contract Assistant") as demo:
    gr.Markdown("# 📜 Smart Contract Assistant & Auditor")
    
    with gr.Tabs():
        # Tab 1: Combined Upload & Chat
        with gr.TabItem("💬 Contract Chat"):
            with gr.Row():
                # Sidebar: Upload Section
                with gr.Column(scale=1):
                    gr.Markdown("### 📁 Document Upload")
                    gr.Markdown("Upload your contract (PDF/DOCX) to train the AI.")
                    file_input = gr.File(label="Select File", file_types=[".pdf", ".docx"])
                    upload_btn = gr.Button("🚀 Process & Train", variant="primary")
                    upload_status = gr.Markdown("*Status: Waiting for document*")
                
                # Main: Chat Section
                with gr.Column(scale=2):
                    gr.Markdown("### 💬 Chat Interface")
                    output_text = gr.Textbox(label="AI Response", interactive=False, lines=10)
                    input_text = gr.Textbox(
                        label="Your Question", 
                        placeholder="e.g., What are the payment terms in this contract?",
                        lines=2
                    )
                    submit_btn = gr.Button("Send Message", variant="primary")

            # Click Events for Chat Tab
            upload_btn.click(fn=handle_file_upload, inputs=file_input, outputs=upload_status)
            submit_btn.click(fn=ask_bot, inputs=input_text, outputs=output_text)

        # Tab 2: Audit Section
        with gr.TabItem("⚖️ Automated Audit"):
            gr.Markdown("### ⚖️ RAG Performance Audit")
            gr.Markdown("This tool generates synthetic questions to stress-test the AI's accuracy on the uploaded document.")
            
            eval_btn = gr.Button("🔍 Run Accuracy Audit", variant="secondary")
            score_display = gr.Markdown("Click the button to analyze system performance.")
            
            results_table = gr.Dataframe(
                headers=["Status", "Question", "Score", "Justification"],
                datatype=["str", "str", "str", "str"],
                wrap=True
            )
            
            eval_btn.click(fn=run_eval_ui, inputs=None, outputs=[score_display, results_table])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)