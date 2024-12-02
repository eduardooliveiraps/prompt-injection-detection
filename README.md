# Prompt Injection Detection

This project focuses on detecting prompt injection attacks using embedding-based machine learning classifiers.


## Getting Started

Please copy .env.example to .env and fill in the required fields.
```bash
cp .env.example .env
```

To get the model-artifacts, you could download `models.zip` from [here](https://drive.google.com/drive/folders/1_-e51Ax3c3vW6HhbKCS4a2G0egb69rqy?usp=sharing) and extract it to the root directory of the project.

To get the dataset, you could download `data.zip` from [here](https://drive.google.com/drive/folders/1_-e51Ax3c3vW6HhbKCS4a2G0egb69rqy?usp=sharing) and extract it to the root directory of the project.


To intsall llm-guard, run the following command:
```bash
make install-llm-guard
```

To install dev requirements, run the following command:
```bash
make install-dev
# or pip install -r requirements.txt
```

## Notebooks

The notebooks are located in the `notebooks` directory. Each notebook has a specific purpose and is named accordingly.

- `data-preprocessing.ipynb`: This notebook is used to preprocess the dataset, it outputs a sample of the dataset to be used later on.
- `generate-embeddings.ipynb`: This notebook is used to generate embeddings for the dataset.
- `simple_model_training.ipynb`: This notebook is used to train a simple model on the dataset and evaluate it.
- `colab_model_training.ipynb`: This notebook is used to get a pre-trained model from huggingface and train  the dataset using Google Colab and evaluate it.
- `llm_guard_simulation.py`: This notebook is used to simulate the LLM-Guard model on different models and see the results.
  

## LLM Guard

LLM Guard is a lib located in the `llm_guard` directory. It is used to act as a wrapper for the LLM model and apply some preprocessing and postprocessing on the user input to ensure that the model is not being attacked by prompt injection attacks.


## LLM Guard: Streamlit App

Before running the streamlit app, make sure to install the required dependencies by running the following command:
```bash
pip install -r streamlit-app/requirements.txt
```

To run the streamlit app, run the following command:
```bash
make st-run
```
