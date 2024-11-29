from setuptools import setup, find_packages

setup(
    name="llm_guard",
    version="0.1.0",
    author="Seif Mostafa",
    author_email="your.email@example.com",
    description="A short description of your package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(
        include=["llm_guard"]
    ),  # Automatically finds all packages in the project directory
    install_requires=[
        "openai==1.54.3",
        "python-dotenv==1.0.1",
        "litellm==1.53.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
