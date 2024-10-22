import streamlit as st

# from prompts_en import (
from prompts_nl import (
    default_system_prompt,
    default_task_prompt,
    default_formatting_instruction,
)
from configuration import (
    MAX_OUTPUT_TOKENS,
    TEMPERATURE,
    TOP_P,
    TOP_K,
)
from reply import reply

# Set Debug mode On/Off here
DEBUG=True


_CONVERSATION_HISTORY = "history"

with st.sidebar:
    system_prompt = st.text_area(
        "System Prompt (Defines the behavior of the assistant)",
        height=3,
        value=default_system_prompt,
    )
    task_prompt = st.text_area("Task Prompt", height=3, value=default_task_prompt)
    formatting_instruction = st.text_area(
        "Formatting Instruction", height=3, value=default_formatting_instruction
    )
    max_output_tokens = st.slider(
        "Max Output Tokens", min_value=10, max_value=5000, value=MAX_OUTPUT_TOKENS
    )
    temperature = st.slider(
        "Temperature", min_value=0.0, max_value=1.0, value=TEMPERATURE
    )
    top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=TOP_P)
    top_k = st.slider("Top K", min_value=0, max_value=100, value=TOP_K)


if _CONVERSATION_HISTORY not in st.session_state:
    st.session_state[_CONVERSATION_HISTORY] = []


def append_msg(msg, sender, chunks=None):
    st.session_state[_CONVERSATION_HISTORY].append(
        {"name": sender, "text": msg, "chunks": chunks}
    )


avatar = {"user": "🤓", "assistant": "🦜"}


history = st.session_state[_CONVERSATION_HISTORY]

if message := st.chat_input("Type your message here..."):
    append_msg(message, "user")
    text, chunks = reply(
        history=history,
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        formatting_instruction=formatting_instruction,
        generation_params={
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        },
    )
    append_msg(text, "assistant", chunks)


button_pop = st.sidebar.button("Pop last message")
if button_pop:
    ans = st.session_state[_CONVERSATION_HISTORY].pop()
    while ans["name"] == "assistant":
        ans = st.session_state[_CONVERSATION_HISTORY].pop()
    print(ans["text"])
    message = ans["text"]

button_save_current_params = st.sidebar.button("Save current parameters")
if button_save_current_params:
    import json
    import datetime
    import os
    os.makedirs(os.path.join(os.getcwd(), "data"), exist_ok=True)

    dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    doc = {
        "system_prompt": system_prompt,
        "task_prompt": task_prompt,
        "formatting_instruction": formatting_instruction,
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }

    if DEBUG: # save also the conversation history
        doc["message"] = message
        doc["history"] = history
        
    with open(os.path.join(os.getcwd(), "data", f"current_params_{dt}.json"), "w") as f:
        f.write(json.dumps(doc))
        st.sidebar.success("Parameters saved")


button_clear = st.sidebar.button("Clear Conversation History")
if button_clear:
    st.session_state[_CONVERSATION_HISTORY].clear()
    st.cache_data.clear()        


for msg in history:
    with st.chat_message(msg["name"], avatar=avatar[msg["name"]]):
        st.markdown(msg["text"])
