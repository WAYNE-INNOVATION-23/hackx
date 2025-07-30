from flask import Flask, request, jsonify
import requests
import fitz  # PyMuPDF
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)

# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

@app.route('/hackrx/run', methods=['POST'])
def run():
    try:
        # Validate Authorization header
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({'error': 'Missing or invalid Authorization header'}), 401

        # Load request data
        data = request.get_json()
        pdf_url = data.get("documents")
        questions = data.get("questions", [])

        if not pdf_url or not questions:
            return jsonify({'error': 'Missing PDF URL or questions'}), 400

        # Download the PDF
        response = requests.get(pdf_url)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to download PDF'}), 400

        pdf_path = "temp.pdf"
        with open(pdf_path, 'wb') as f:
            f.write(response.content)

        # Extract text from PDF
        doc = fitz.open(pdf_path)
        full_text = "\n".join([page.get_text() for page in doc])
        doc.close()

        # Initialize Gemini model
        model_gemini = genai.GenerativeModel('gemini-pro')

        # Process questions
        answers = []
        for question in questions:
            prompt = f"""Based on the following insurance policy document, answer the question:

Document:
{full_text}

Question: {question}
Answer:"""

            gemini_response = model_gemini.generate_content(prompt)
            answers.append(gemini_response.text.strip())

        return jsonify({'answers': answers})

    except Exception as e:
        return jsonify({'error': f'PDF processing failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
