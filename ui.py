import streamlit as st
import base64

# Import your RAG backend
# Adjust this import based on your project structure
from src.pipeline.ui_pipeline import run_rag_for_ui_pipeline

st.set_page_config(page_title="ResearchGPT", page_icon="📄", layout="centered")

# Function to convert local image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Use the path to your specific image
img_base64 = get_base64_image("C:/Users/vvrag/UROP bg pic/Background(neew).png") 

# --- CUSTOM CSS ---
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{img_base64}");
        background-attachment: fixed;
        background-size: cover;
        color: #ffffff;
    }}

    /* The Literature Review Box */
    .review-box {{
        background-color: rgba(255, 255, 255, 0.07); 
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        padding: 30px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
        margin-top: 20px;
        margin-bottom: 20px;
        color: #ffffff;
        line-height: 1.6;
    }}

    /* Styling headings inside the review box */
    .review-box h1, .review-box h2, .review-box h3 {{
        color: #A4ADFF !important; /* Soft blue-lilac for headers */
        margin-top: 15px;
    }}

    h1, h2, h3, span, label {{
        color: #ffffff !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }}

    .stTextArea textarea {{
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }}

    .stButton>button {{
        background-color: #4D56A6;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1rem;
        font-weight: bold;
    }}
    </style>
    """, unsafe_allow_html=True)

st.title("📄 ResearchGPT")
st.caption("A Retrieval-Augmented Research Assistant grounded in academic papers")

st.divider()

st.subheader("Ask a Research Question")

query = st.text_area(
    "Enter your question",
    placeholder="What methodologies are used for graph-based traffic prediction?",
    height=120
)

strategy = st.selectbox(
    "Retrieval Strategy",
    ["hyde", "complex", "standard"]
)

if st.button("Analyze Papers", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a research question.")
    else:
        with st.spinner("Running RAG pipeline..."):
            result = run_rag_for_ui_pipeline(query, strategy)

        st.subheader("Generated Answer")
        st.write(result['review'])

        st.info(
            f"Strategy: {result['strategy']} | "
            f"Papers processed: {result['num_papers']}"
        )

st.divider()

st.caption(
    "Responses are generated using Retrieval-Augmented Generation (RAG) "
    "to ensure grounding in academic literature."
)

