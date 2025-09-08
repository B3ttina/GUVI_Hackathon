import os
from transformers import pipeline
import fitz  # PyMuPDF
from pypdf import PdfReader
import markdown
from bs4 import BeautifulSoup

# -----------------------------
# Configuration
# -----------------------------
OUTPUT_FOLDER = "../parsed_texts"  # Folder to save parsed text
MAX_CHARS_PER_CHUNK = 1000         # For chunking long documents

# -----------------------------
# 1. Document Parsing Functions
# -----------------------------
def parse_pdf_pymupdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text().replace("\n", " ")
    return text.strip()

def parse_pdf_pypdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for i, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text:
            text += f"\n--- Page {i} ---\n{page_text}"
        else:
            text += f"\n--- Page {i} ---\n[No extractable text]\n"
    return text

def parse_markdown(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    html = markdown.markdown(md_text)
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ").strip()

def parse_html(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        html_text = f.read()
    soup = BeautifulSoup(html_text, "html.parser")
    return soup.get_text(separator=" ").strip()

def detect_file_type(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return "pdf"
    elif ext in [".md", ".markdown"]:
        return "markdown"
    elif ext in [".html", ".htm"]:
        return "html"
    else:
        return None

def save_text(text, filename="output.txt"):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    out_path = os.path.join(OUTPUT_FOLDER, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ Saved parsed text to {out_path}")

# -----------------------------
# 2. LLM Integration Functions
# -----------------------------
print("⚡ Loading Hugging Face generative QA model...")
gen_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

def chunk_text(text, max_chars=MAX_CHARS_PER_CHUNK):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def generate_answer_from_text(question, text):
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
    for ans in answers:
        if ans.strip():
            return ans.strip()
    return "Sorry, I couldn't generate an answer from the document."

def generate_answer_from_file(question, file_path):
    file_type = detect_file_type(file_path)
    if file_type == "pdf":
        text = parse_pdf_pymupdf(file_path)
    elif file_type == "markdown":
        text = parse_markdown(file_path)
    elif file_type == "html":
        text = parse_html(file_path)
    else:
        raise ValueError("Unsupported file type for QA!")
    return generate_answer_from_text(question, text)

# -----------------------------
# 3. Main Execution
# -----------------------------
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    samples_dir = os.path.join(base_dir, "..", "samples")
    
    # Put any test file here (PDF, Markdown, HTML)
    file_name = "sample.pdf"  
    file_path = os.path.join(samples_dir, file_name)

    print("🔍 Looking for:", file_path)
    print("📂 Exists?", os.path.exists(file_path))

    try:
        # Parse & save text
        file_type = detect_file_type(file_path)
        if file_type == "pdf":
            parsed_text = parse_pdf_pymupdf(file_path)
        elif file_type == "markdown":
            parsed_text = parse_markdown(file_path)
        elif file_type == "html":
            parsed_text = parse_html(file_path)
        else:
            raise ValueError("Unsupported file type!")
        
        print("✅ Parsed text length:", len(parsed_text))
        save_text(parsed_text, f"{os.path.splitext(file_name)[0]}_parsed.txt")

        # Example QA
        question = "What is the goal of supervised learning?"
        answer = generate_answer_from_text(question, parsed_text)
        print("\n💬 Example QA")
        print("Q:", question)
        print("A:", answer)

    except Exception as e:
        print("❌ Error:", e)
