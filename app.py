import streamlit as st

st.set_page_config(
    page_title="AI Question Generator",
    page_icon="✨",
    layout="wide"
)
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
import pdfplumber
import docx

# ======================================================
# AESTHETIC CSS (PREMIUM UI)
# ======================================================
st.markdown("""
<style>

body {
    background: #F0F3F9 !important;
}

/* Full width layout */
.main > div {
    max-width: 900px;
    margin: auto;
}

/* HEADER STYLE */
.header-box {
    background: linear-gradient(135deg, #6A82FB, #FC5C7D);
    padding: 35px;
    border-radius: 22px;
    text-align: center;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0px 8px 18px rgba(0,0,0,0.12);
}

.header-title {
    font-size: 40px;
    font-weight: 900;
    margin-bottom: -5px;
}
.header-sub {
    font-size: 16px;
    opacity: 0.95;
}

/* CARD CLEAN STYLE */
.clean-card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.10);
    margin-bottom: 20px;
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(135deg, #6A82FB, #5A53E0);
    width: 100%;
    height: 52px;
    color: white;
    border-radius: 12px;
    font-size: 18px;
    font-weight: 700;
    border: none;
    transition: 0.2s;
}
.stButton>button:hover {
    transform: scale(1.03);
    background: linear-gradient(135deg, #5A53E0, #6A82FB);
}

/* TEXT AREA */
textarea {
    background: white !important;
    border-radius: 12px !important;
    padding: 15px !important;
    font-size: 16px !important;
    border: 1px solid #DDE3F0 !important;
}

/* RESULT QUESTION BUBBLE */
.result-bubble {
    background: linear-gradient(135deg, #EEF3FF, #FAFBFF);
    padding: 15px;
    border-radius: 18px;
    margin-bottom: 12px;
    font-size: 17px;
    border-left: 6px solid #6A82FB;
    display: flex;
    gap: 12px;
    align-items: flex-start;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.06);
}
.bubble-icon {
    font-size: 20px;
    margin-top: 3px;
}
            

</style>
""", unsafe_allow_html=True)

# ======================================================
# FILE EXTRACT FUNCTION
# ======================================================
def extract_text(file):
    if file.type == "text/plain":
        return file.read().decode("utf-8")

    elif file.type == "application/pdf":
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        return text

    elif file.type.endswith("document"):
        doc_file = docx.Document(file)
        return "\n".join([p.text for p in doc_file.paragraphs])

    return None


# ======================================================
# LOAD MODEL
# ======================================================
@st.cache_resource
def load_model():
    model_name = "citraulia/t5-question-generator"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to("cpu")
    return tokenizer, model

tokenizer, model = load_model()

# ======================================================
# QUESTION GENERATOR
# ======================================================
def generate_questions(context, answer, max_length=72, num_questions=5):
    input_text = f"generate question: {context} <hl> {answer} <hl>"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    outputs = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        do_sample=True,
        top_k=120,
        top_p=0.92,
        temperature=1.2,
        repetition_penalty=2.0,
        num_return_sequences=num_questions
    )

    final_q = []
    for o in outputs:
        q = tokenizer.decode(o, skip_special_tokens=True)
        if q not in final_q:
            final_q.append(q)
    return final_q


# ======================================================
# HEADER WEBSITE
# ======================================================
st.markdown("""
<div class='header-box'>
    <div class='header-title'>✨ AI Question Generator</div>
    <div class='header-sub'>Generate intelligent questions instantly from any text or uploaded document.</div>
</div>
""", unsafe_allow_html=True)

# ======================================================
# INPUT AREA
# ======================================================
st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
st.subheader("📄 Upload File or Paste Text")

uploaded_file = st.file_uploader("Upload PDF / TXT / DOCX", type=["pdf", "txt", "docx"])

if uploaded_file:
    extracted = extract_text(uploaded_file)
    if extracted:
        st.success("File successfully extracted!")
        context = st.text_area("Extracted Text", extracted, height=200)
    else:
        st.error("Failed to extract text.")
        context = ""
else:
    context = st.text_area("Context (paragraph)", height=180)

answer = st.text_input("Answer (taken from context)")
st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# NUMBER OF QUESTIONS
# ======================================================
st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
num_q = st.number_input("Number of questions", min_value=1, max_value=10, value=5)
st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# GENERATE OUTPUT
# ======================================================
if st.button("🚀 Generate Questions"):
    if context and answer:
        with st.spinner("Generating questions..."):
            questions = generate_questions(context, answer, num_questions=num_q)

        st.subheader("🧠 Generated Questions")

        for i, q in enumerate(questions, 1):
            st.markdown(
                f"""
                <div class='result-bubble'>
                    <div class='bubble-icon'>💬</div>
                    <div><b>{i}. </b>{q}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.warning("Please provide both a context and an answer!")