from setuptools import setup, find_packages

setup(
    name="LLM-State-Machine",
    version="0.1.0",
    description="A Python framework for FSM-based LLM agents",
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
        "License :: OSI Approved :: Apache 2.0 License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
