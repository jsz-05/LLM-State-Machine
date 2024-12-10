from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="fsm-llm",
    version="0.1.3",
    description="A Python framework for creating FSM-based LLM agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jeffrey Zhou",
    author_email="jeffreyzhou3@outlook.com",
    url="https://github.com/jsz-05/LLM-State-Machine",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "openai>=0.27.0",
        "pydantic>=1.10.0",
        "jinja2>=3.0.0",
        "python-dotenv>=0.15.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
