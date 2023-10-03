import streamlit as st
from reply import reply

_CONVERSATION_HISTORY = "history"

if _CONVERSATION_HISTORY not in st.session_state:
    st.session_state[_CONVERSATION_HISTORY] = []

def append_msg(msg,sender,chunks=None):
    st.session_state[_CONVERSATION_HISTORY].append({
        "name": sender,
        "text": msg,
        "chunks": chunks
    })

avatar = {
    "user": "🤓",
    "assistant": "🦜"
}


history = st.session_state[_CONVERSATION_HISTORY]
message = st.chat_input("say something...")
if message:
    append_msg(message,"user")
    text,chunks = reply(history)
    append_msg(text, "assistant", chunks)

for msg in history:
    with st.chat_message(msg["name"], avatar=avatar[msg["name"]]):
        st.write(msg["text"])

    

