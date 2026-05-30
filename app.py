import streamlit as st

from backend import answer_question, process_uploaded_file


st.set_page_config(
    page_title="Local RAG Chat",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Local RAG Chat")


uploaded_file = st.sidebar.file_uploader(
    "Upload PDF or TXT",
    type=["pdf", "txt"],
)

if uploaded_file:
    with st.spinner("Processing document..."):
        process_uploaded_file(uploaded_file)

    st.sidebar.success("Document uploaded successfully!")


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Upload a document and ask questions 📄",
        }
    ]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input("Ask anything...")

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        try:
            full_response = answer_question(question)
            message_placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"Error: {str(e)}"
            st.error(full_response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response,
        }
    )
