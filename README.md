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