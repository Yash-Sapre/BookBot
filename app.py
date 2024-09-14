import streamlit as st
from llm import respond
from embeddings import extract_doc,convert_to_chunks,embed_doc,semantic_search,reset_doc

if "uploader_visible" not in st.session_state:
    st.session_state["uploader_visible"] = True

if st.session_state["uploader_visible"]:
    file = st.file_uploader("Upload your file")
    if file:
        reset_doc()
        extract_doc(file)
        convert_to_chunks()
        embed_doc()


if "messages" not in st.session_state:
    st.session_state.messages = []




for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
    
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    data = semantic_search(prompt)
    resp = respond(prompt,data)
    with st.chat_message("bot"):
        st.markdown(f"{resp}")
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "bot", "content": resp}) 