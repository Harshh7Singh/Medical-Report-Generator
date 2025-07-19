# Medical-Report-Generator
 Upload a medical report → Generate a patient-friendly explanation of the content.
# medical_summary_generator.py
# Upload a medical report → Generate a patient-friendly explanation
!pip install -U langchain langchain-community langchain-openai sentence-transformers faiss-cpu gradio PyMuPDF --quiet
import os
import gradio as gr
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# 🔐 API key setup (replace before running)
os.environ["OPENAI_API_KEY"] = "sk-REPLACE_YOUR_KEY"

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load LLM (Groq API)
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["OPENAI_API_KEY"],
    model="llama3-70b-8192",
    temperature=0.2
)

# Globals for chain and vectorstore
vectordb = None
qa_chain = None

def load_medical_pdf(file):
    global vectordb, qa_chain

    loader = PyMuPDFLoader(file.name)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    vectordb = FAISS.from_documents(chunks, embedding_model)
    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    return f"✅ Processed {len(chunks)} chunks. Ask your health question below."

def explain_to_patient(question):
    if not qa_chain:
        return "⚠️ Please upload a medical document first."
    try:
        return qa_chain.run(f"Explain this in simple terms for a patient: {question}")
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Gradio Interface
with gr.Blocks(title="🧠 Medical Summary Generator") as demo:
    gr.Markdown("# 🩺 Medical Report → Patient-Friendly Summary")
    gr.Markdown("Upload clinical notes or discharge summaries. Ask in plain English.")

    file_input = gr.File(label="📄 Upload Medical PDF")
    status_output = gr.Textbox(label="System Message", interactive=False)
    file_input.change(fn=load_medical_pdf, inputs=file_input, outputs=status_output)

    question_input = gr.Textbox(label="Ask a Health-Related Question", placeholder="e.g., What was the diagnosis?")
    submit_btn = gr.Button("Generate Summary")
    summary_output = gr.Textbox(label="Layman Explanation", lines=6)

    submit_btn.click(fn=explain_to_patient, inputs=question_input, outputs=summary_output)

demo.launch(debug=True, share=True)
