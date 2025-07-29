import os
import tempfile
import requests
import pdfplumber
from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-1.5-flash")

# Flask app
app = Flask(__name__)

# Helper: download and read PDF from URL
def extract_text_from_pdf_url(url):
    response = requests.get(url)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name

    full_text = ""
    with pdfplumber.open(tmp_path) as pdf:
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"
    return full_text

# Helper: Create vector DB from text
def build_faiss_index(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return FAISS.from_texts(chunks, embeddings)

# POST endpoint
@app.route("/hackrx/run", methods=["POST"])
def run_hackrx():
    if request.headers.get("Authorization") != f"Bearer {os.getenv('HACKRX_API_KEY')}":
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    url = data.get("documents")
    questions = data.get("questions", [])

    if not url or not questions:
        return jsonify({"error": "Missing 'documents' URL or 'questions' list"}), 400

    try:
        pdf_text = extract_text_from_pdf_url(url)
        db = build_faiss_index(pdf_text)

        answers = []
        for q in questions:
            docs = db.similarity_search(q, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = f"""
You are a healthcare policy expert helping users understand their insurance documents.

Context:
{context}

Question:
{q}

Instructions:
- Provide a clear, concise answer.
- If the information is not available in the context, say "Not specified in the document."
- Avoid assumptions or generalizations.
- Keep the answer within 1-2 sentences.
- Respond only with the answer to the question.

Answer:"""

            response = model.generate_content(prompt)
            answers.append(response.text.strip())

        return jsonify({"answers": answers})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Local test only
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
