import json
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig


st.set_page_config(page_title="360智脑大模型")
st.title("360智脑大模型")

MODEL_NAME_OR_PATH = "qihoo360/360Zhinao-7B-Chat-4K"

@st.cache_resource
def load_model_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH,
        use_fast=False,
        trust_remote_code=True
    )
    generation_config = GenerationConfig.from_pretrained(MODEL_NAME_OR_PATH)
    return model, tokenizer, generation_config


def clear_chat_messages():
    del st.session_state.messages


def init_chat_messages():
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown("您好，我是360智脑大模型，很高兴为您服务😄")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []
    
    return st.session_state.messages


max_new_tokens = st.sidebar.slider("max_new_tokens", 0, 2048, 512, step=1)
top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.8, step=0.01)
top_k = st.sidebar.slider("top_k", 0, 100, 50, step=1)
temperature = st.sidebar.slider("temperature", 0.0, 2.0, 1.0, step=0.01)
do_sample = st.sidebar.checkbox("do_sample", value=True)

def main():
    model, tokenizer, generation_config = load_model_tokenizer()
    messages = init_chat_messages()

    if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
        with st.chat_message("user", avatar='🧑‍💻'):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar='🤖'):
            placeholder = st.empty()
            messages.append({"role": "user", "content": prompt})

            generation_config.max_new_tokens = max_new_tokens
            generation_config.top_p = top_p
            generation_config.top_k = top_k
            generation_config.temperature = temperature
            generation_config.do_sample = do_sample
            print("generation_config: ", generation_config)

            for response in model.chat(tokenizer=tokenizer, messages=messages, stream=True, generation_config=generation_config):
                placeholder.markdown(response)
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        
        messages.append({"role": "assistant", "content": response})
        print("messages: ", json.dumps(messages, ensure_ascii=False), flush=True)

    st.button("清空对话", on_click=clear_chat_messages)


if __name__ == "__main__":
    main()