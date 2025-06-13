import streamlit as st
import requests
import pandas as pd

BACKEND_URL = "http://localhost:8000/ask"  # FastAPI endpoint

# Page Configuration
st.set_page_config(
    page_icon="🎓", 
    page_title="Ask the Scholar", 
    initial_sidebar_state="auto",
    layout="wide")

# Load external CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🎓 Ask the Scholar")
st.sidebar.markdown("""
    **About**  
    This is a multimodal RAG-based assistant that allows you to interact with research papers using natural language.   
""")
st.sidebar.info("""
    **Features**  
    - Chat with research papers
    - Get relevant answers
    - Get multimodal responses
    - Interactive chat interface
""")

# Main title with an icon
st.markdown(
    """
    <div class="custom-header"'>
        <span>👨🏻‍🏫 Ask the Scholar</span><br>
        <span>A Multimodal RAG-based assistant to Chat with Research Papers</span>
    </div>
    """,
    unsafe_allow_html=True
)

# Horizontal line
st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)

# Initialize Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# # Display Welcome Message
# if st.session_state.welcome_message == True:
#     st.success("""
#         **Welcome to Ask the Scholar!**
#         - I'll help you find answers in research papers.
#         - Just type your question below and I'll do my best to assist you.
#     """)

# Chat History Display
chat_container = st.container()
with chat_container:
    for chat in st.session_state.chat_history:
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"<div class='user-msg'>{chat['user']}</div>", unsafe_allow_html=True)
        with st.chat_message("assistant", avatar="👨🏻‍🏫"):
            # st.markdown(f"<div class='assistant-msg'>{chat['assistant']}</div>", unsafe_allow_html=True)
            st.markdown(chat['assistant'])

# User Input
user_input = st.chat_input("Ask a question...")

if user_input:
    with st.chat_message("user", avatar="👤"):
        st.markdown(f"<div class='user-msg'>{user_input}</div>", unsafe_allow_html=True)

    with st.spinner("Retrieving information..."):
        try:
            response = requests.post(BACKEND_URL, json={"query": user_input})
            response.raise_for_status()
            result_data = response.json()
            result = result_data.get("response", "")
            images = result_data.get("images", [])
            tables = result_data.get("tables", [])
        except Exception:
            result = "❌ Error: Something went wrong."
            images, tables = [], []

    with st.chat_message("assistant", avatar="👨🏻‍🏫"):
        # st.markdown(f"<div class='assistant-msg'>{result}</div>", unsafe_allow_html=True)
        st.markdown(result)

        try:
            if images:
                st.markdown("**🔍 Relevant Images:**")
                for img_path in images:
                    st.image(img_path, use_column_width=True)

            if tables:
                st.markdown("**📊 Relevant Tables:**")
                for table_path in tables:
                    try:
                        df = pd.read_csv(table_path)
                        st.dataframe(df)
                    except Exception as e:
                        st.warning(f"Could not display table {table_path}: {e}")
        except Exception:
            st.error("Could not display results. Please try again.")

    # Update session history
    st.session_state.chat_history.append({
        "user": user_input,
        "assistant": result
    })
