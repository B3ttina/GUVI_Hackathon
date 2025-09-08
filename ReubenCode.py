import os
import sys
print("Running from:", __file__)
print("Python path:", sys.path)

from transformers import pipeline
from PyPDF2 import PdfReader

# -----------------------------
# 1. Load Hugging Face Generative QA model
# -----------------------------
gen_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

# -----------------------------
# 2. Extract text from PDF
# -----------------------------
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# -----------------------------
# 3. Split text into smaller chunks
# -----------------------------
def chunk_text(text, max_chars=1000):
    """
    Splits text into chunks (default: 1000 characters).
    Useful because models have input length limits.
    """
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

# -----------------------------
# 4. Ask a question over the chunks (Generative)
# -----------------------------
def generate_answer_from_pdf(question, pdf_path):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    answers = []
    for chunk in chunks:
        prompt = f"Answer the question based on the following context:\n{chunk}\n\nQuestion: {question}\nAnswer:"
        try:
            result = gen_pipeline(prompt, max_new_tokens=200, truncation=True)
            answers.append(result[0]['generated_text'])
        except Exception as e:
            print("Error:", e)
            continue

    # Return the most relevant non-empty answer
    for ans in answers:
        if ans.strip():
            return ans.strip()

    return "Sorry, I couldn't generate an answer from the document."

# -----------------------------
# 5. Run the system
# -----------------------------
if _name_ == "_main_":
    pdf_path = "C:\\Users\\sunis\\OneDrive\\Desktop\\Hackathon_Project\\LLM\\sample.pdf"   
    question = "What is the goal of supervised learning ?"

    answer = generate_answer_from_pdf(question, pdf_path)
    print("Q:", question)
    print("Answer:", answer)
