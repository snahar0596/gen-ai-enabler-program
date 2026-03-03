import sys
import os
sys.path.append(os.path.abspath("smart-cpg-decision-agent"))

import pandas as pd
from src.agent.agent_core import create_cpg_agent

# Use a mock or rely on OPENAI_API_KEY if set.
from langchain_core.language_models import FakeListLLM

os.environ["OPENAI_API_KEY"] = "fake"

df = pd.DataFrame({'date': ['2022-01-01'], 'sku_id': [101], 'category': ['Beverages'], 'units_sold': [18], 'revenue': [90.0], 'promo_flag': [0], 'price': [5.0], 'inventory_level': [550]})

# Override the LLM in create_cpg_agent for testing
import src.genai.llm_interface
src.genai.llm_interface.get_llm = lambda: FakeListLLM(responses=["I am a fake AI response"])

agent = create_cpg_agent(df)
try:
    result = agent.invoke({"input": "hello"})
    print("OUTPUT KEY:", result.get("output"))
except Exception as e:
    print(f"Error: {e}")
