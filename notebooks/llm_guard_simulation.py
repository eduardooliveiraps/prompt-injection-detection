# %%
import os
import random
import joblib
import logging

from pathlib import Path


from openai.resources.chat import Completions
from openai.resources.chat.completions import ChatCompletion
from openai import OpenAI
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from sentence_transformers import SentenceTransformer


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

from dotenv import load_dotenv

from llm_guard import LLMGaurd

# %%
# define constants

# load environment variables
filepath = Path(__file__)
env_filepath = filepath.parent / ".env"
load_dotenv(dotenv_path=env_filepath)

if not os.environ.get("OPENAI_APIKEY"):
    raise ValueError("OPENAI_APIKEY not found in the environment variables")

# get the api key and create the openai object
api_key = os.environ.get("OPENAI_APIKEY")

openai = OpenAI(
    api_key=api_key,
)
completions = Completions(client=openai)


# %%
# define helper functions
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


# %%
# define LLM responses
invalid_input_response = create_invalid_response("Invalid input")
invalid_output_response = create_invalid_response("Invalid output")


# llm_guard = LLMGaurd(completions=completions)

# llm_guard with random response
llm_guard = LLMGaurd(
    completions=completions,
    input_guard=lambda _: random.choice([True, False]),
    output_guard=lambda _: random.choice([True, False]),
    invalid_input_response=invalid_input_response,
    invalid_output_response=invalid_output_response,
)

# %%
# Simulate a conversation with the LLMGuard and Random model

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Hello, my name is"},
        ],
    }
]

for _ in range(5):
    response = llm_guard.create(model="gpt-4o-mini", messages=messages, max_tokens=10)
    print(response.choices[0].message.content)

# %%
# Simulate a conversation with the LLMGuard and a LGB model


def get_txt_embedding_for_lgb_model(text: str):
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    text = text.replace("\n", " ")
    try:
        return embedding_model.encode(text, show_progress_bar=False).tolist()
    except Exception as e:
        logging.error("Error encoding text: %s", e)
        return None


def get_lgb_model(model_name: str):
    model_path = f"../models/{model_name}"
    if not Path(model_path).exists():
        raise FileNotFoundError("Model not found")
    return joblib.load(model_path)


def lgb_model_input_guard():
    pass


# %%
# Simulate a conversation with the LLMGuard and distilbert/distilroberta-base from the colab notebook
# %%
