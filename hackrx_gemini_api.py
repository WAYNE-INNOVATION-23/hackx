import os
import requests
import fitz  # PyMuPDF
from flask import Flask, request, jsonify
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai

# ✅ Load environment variables
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-1.5-flash")

app = Flask(__name__)

# ✅ Function to load PDF from URL (including Azure Blob)
def load_pdf_from_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError("Failed to download PDF from URL")
    
    with open("temp.pdf", "wb") as f:
        f.write(response.content)

    doc = fitz.open("temp.pdf")
    full_text = "\n".join([page.get_text() for page in doc])
    doc.close()
    return full_text

# ✅ Text splitting + vector indexing
def build_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    db = FAISS.from_texts(chunks, embedding=embeddings)
    return db

@app.route("/hackrx/run", methods=["POST"])
def hackrx_run():
    if request.headers.get("Authorization") != f"Bearer {os.getenv('HACKRX_API_KEY')}":
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    url = data.get("documents")
    questions = data.get("questions", [])
    
    try:
        pdf_text = load_pdf_from_url(url)
        db = build_vectorstore(pdf_text)
    except Exception as e:
        return jsonify({"error": f"PDF processing failed: {str(e)}"}), 400

    answers = []
    for question in questions:
        docs = db.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"""
You are a healthcare policy expert helping users understand their insurance documents.

Context:
{context}

Question:
{question}

Instructions:
- Provide a clear, concise answer.
- If the information is not available in the context, say "Not specified in the document."
- Avoid assumptions or generalizations.
- Keep the answer within 1-2 sentences.
- Respond only with the answer to the question.

Answer:"""

        try:
            response = model.generate_content(prompt)
            answers.append(response.text.strip())
        except Exception as e:
            answers.append("Error: " + str(e))

    return jsonify({"answers": answers})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
