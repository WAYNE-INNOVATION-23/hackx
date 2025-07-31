from flask import Flask, request, jsonify
import requests
import fitz  # PyMuPDF
import google.generativeai as genai
import os

app = Flask(__name__)

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Gemini model
gemini_model = genai.GenerativeModel("models/gemini-1.5-flash"")

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

        # Extract text using PyMuPDF
        doc = fitz.open("temp.pdf")
        full_text = "\n".join([page.get_text() for page in doc])
        doc.close()

        if not full_text.strip():
            return jsonify({'error': 'No text extracted from PDF'}), 400

        # Answer each question using Gemini
        answers = []
        for question in questions:
            prompt = f"Answer the question based on the insurance policy document below:\n\n{full_text}\n\nQuestion: {question}\nAnswer:"
            try:
                result = gemini_model.generate_content(prompt)
                answer = result.text.strip()
            except Exception as e:
                answer = f"Error generating answer: {str(e)}"
            answers.append(answer)

        return jsonify({"answers": answers})

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# Required for Render
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
