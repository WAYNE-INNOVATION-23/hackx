from flask import Flask, request, jsonify
import requests
import fitz  # PyMuPDF
import google.generativeai as genai
import os
import time

app = Flask(__name__)

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Gemini model
gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")

# Function to extract relevant chunks based on keywords in the question
def get_relevant_chunks(text, question, max_len=2000):
    chunks = text.split('\n\n')
    question_keywords = set(question.lower().split())

    filtered_chunks = []
    total_length = 0
    for chunk in chunks:
        if any(word in chunk.lower() for word in question_keywords):
            filtered_chunks.append(chunk.strip())
            total_length += len(chunk)
            if total_length >= max_len:
                break

    if not filtered_chunks:
        return text[:max_len]  # fallback to start of doc if no matches
    return "\n".join(filtered_chunks)

@app.route('/hackrx/run', methods=['POST'])
def run():
    try:
        # Auth header check
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({'error': 'Missing or invalid Authorization header'}), 401

        data = request.get_json()
        pdf_url = data.get("documents")
        questions = data.get("questions", [])

        if not pdf_url or not questions:
            return jsonify({'error': 'Missing PDF URL or questions'}), 400

        # Download PDF
        pdf_response = requests.get(pdf_url, timeout=15)
        if pdf_response.status_code != 200:
            return jsonify({'error': f'Failed to download PDF from URL: {pdf_url}'}), 400

        with open("temp.pdf", "wb") as f:
            f.write(pdf_response.content)

        # Extract text using PyMuPDF (limit to first 6 pages)
        doc = fitz.open("temp.pdf")
        full_text = "\n".join([page.get_text() for page in doc[:6]])
        doc.close()

        if not full_text.strip():
            return jsonify({'error': 'No text extracted from PDF'}), 400

        # Start timing
        total_start = time.time()

        # Generate answers
        answers = []
        for question in questions:
            context = get_relevant_chunks(full_text, question)
            prompt = f"""You are a concise assistant. Based only on the insurance document below, answer the question in one or two short sentences. Do not add explanations.

---DOCUMENT---
{context}

---QUESTION---
{question}

---ANSWER---"""
            try:
                result = gemini_model.generate_content(prompt)
                answer = result.text.strip()
            except Exception as e:
                answer = f"Error: {str(e)}"
            answers.append(answer)

        total_end = time.time()
        response_time = round(total_end - total_start, 2)

        return jsonify({
            "answers": "\n".join(answers),
            "total_response_time_sec": response_time
        })

    except Exception as e:
        return jsonify({'error': f"Server error: {str(e)}"}), 500

# For Render
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
