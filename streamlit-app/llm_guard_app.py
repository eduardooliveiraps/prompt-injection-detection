import os

from itertools import zip_longest


import streamlit as st
import torch

from dotenv import load_dotenv
from openai.resources.chat import Completions
from openai.resources.chat.completions import ChatCompletion
from openai import OpenAI
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from llm_guard import LLMGaurd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from streamlit_chat import message

# ------------------------------
# define openai and completions object
# ------------------------------

env_filepath = ".env"

load_dotenv(dotenv_path=env_filepath)

if not os.environ.get("OPENAI_APIKEY"):
    raise ValueError("OPENAI_APIKEY not found in the environment variables")

# get the api key and create the openai object
api_key = os.environ.get("OPENAI_APIKEY")

openai = OpenAI(
    api_key=api_key,
)
completions = Completions(client=openai)


# ------------------------------
# define llm guard object
# ------------------------------


@st.cache_data()
def get_colab_model_and_tokenizer():
    colab_model_path = "./models/colab_models/20_epochs_distilroberta-base"
    tokenizer_path = "distilbert/distilroberta-base"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(colab_model_path)
    return model, tokenizer


model, tokenizer = get_colab_model_and_tokenizer()


def colab_model_input_guard(messages):
    is_valid = True
    for message in messages:
        text = message["content"][0]["text"]
        inputs = tokenizer(text, return_tensors="pt")
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1)
        is_valid &= predicted_class[0] == 0
    return is_valid


def create_invalid_response(text: str) -> ChatCompletion:
    return ChatCompletion(
        id="",
        object="chat.completion",
        created=0,
        model="",
        choices=[
            Choice(
                index=0,
                finish_reason="length",
                message=ChatCompletionMessage(
                    content=text,
                    role="assistant",
                ),
            )
        ],
    )


llm_guard_with_lgb = LLMGaurd(
    completions=completions,
    input_guard=colab_model_input_guard,
    output_guard=lambda _: True,
    invalid_input_response=create_invalid_response("Nice try .. but no"),
    invalid_output_response=create_invalid_response("Invalid output"),
)


# ------------------------------
# define onclick functions
# ------------------------------
def on_send_message():
    user_input = st.session_state.user_input

    response = llm_guard_with_lgb.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_input},
                ],
            }
        ],
        max_tokens=512,
    )

    bot_msg = response.choices[0].message.content

    st.session_state.user_msgs.append(user_input)
    st.session_state.bot_msgs.append(bot_msg)
    st.session_state.user_input = ""


def on_clear_btn_clicked():
    del st.session_state.user_msgs[:]
    st.session_state.bot_msgs = ["Hello, How can I help you?"]
    st.session_state.user_input = ""


st.session_state.setdefault(
    "user_msgs",
    [],
)
st.session_state.setdefault(
    "bot_msgs",
    [
        "Hello, How can I help you?",
    ],
)


# ------------------------------
# Streamlit app layout starts here
# ------------------------------
st.title(":lock: LLM Guard App")

chat_placeholder = st.empty()

with chat_placeholder.container():
    merged_messages = [
        item
        for pair in zip_longest(
            st.session_state["bot_msgs"], st.session_state["user_msgs"]
        )
        for item in pair
        if item is not None
    ]

    for i, msg in enumerate(merged_messages):
        is_user = i % 2 == 1
        message(msg, is_user=is_user, key=f"{i}")

with st.container():
    st.text_input(
        "User input",
        key="user_input",
        placeholder="Type a message...",
    )

    col1, _col2, _col3, col4 = st.columns(4)

    with col1:
        st.button("Send", on_click=on_send_message)

    with col4:
        st.button("Clear Chat", on_click=on_clear_btn_clicked)
