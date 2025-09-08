import streamlit as st
import os
from main import (
    parse_pdf_pymupdf,
    parse_markdown,
    parse_html,
    detect_file_type,
    generate_answer_from_text
)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Document QA App",
    page_icon="📄",
    layout="wide",
)

# -----------------------------
# Header
# -----------------------------
st.title("📄 Document Question Answering App")
st.markdown(
    """
    <p style="font-size:18px;">
    Upload a <b>PDF, Markdown, or HTML document</b>, ask a question, and get 
    <span style="color:#4CAF50;">AI-powered answers</span> instantly!
    </p>
    """,
    unsafe_allow_html=True,
)

st.divider()

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader(
        "📂 Upload your document (PDF, Markdown, HTML)", 
        type=["pdf", "md", "markdown", "html", "htm"]
    )
    question = st.text_area(
        "❓ Ask a question:", 
        placeholder="e.g., What is the main topic of this document?"
    )
    submit_btn = st.button("🔎 Get Answer")

with col2:
    st.subheader("💡 Answer")
    answer_placeholder = st.empty()

# -----------------------------
# Processing
# -----------------------------
if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success(f"✅ {uploaded_file.name} uploaded successfully!")

    if submit_btn and question.strip():
        with st.spinner("Processing your question... ⏳"):
            try:
                # Detect type and parse text
                file_type = detect_file_type(temp_path)
                if file_type == "pdf":
                    parsed_text = parse_pdf_pymupdf(temp_path)
                elif file_type == "markdown":
                    parsed_text = parse_markdown(temp_path)
                elif file_type == "html":
                    parsed_text = parse_html(temp_path)
                else:
                    st.error("Unsupported file type!")
                    parsed_text = ""

                # Generate answer using parsed text
                answer = generate_answer_from_text(question, parsed_text)

                # Show answer in nice container
                answer_placeholder.markdown(
                    f"""
                    <div style="
                        background-color:var(--background-color);
                        padding:15px;
                        border-radius:12px;
                        border:1px solid #ddd;
                        font-size:16px;
                        color:var(--text-color);
                    ">
                        <b>🤖 Answer:</b><br>{answer}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"❌ Error: {e}")

elif uploaded_file is None:
    st.info("👆 Please upload a document to get started.")
