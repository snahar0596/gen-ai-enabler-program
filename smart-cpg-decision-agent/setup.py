from setuptools import setup, find_packages

setup(
    name="smart_cpg_decision_agent",
    version="0.1.0",
    description="A Smart Decision Support Agent for CPG businesses leveraging GenAI and Databricks.",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "langchain",
        "streamlit"
    ],
    python_requires=">=3.8",
)
