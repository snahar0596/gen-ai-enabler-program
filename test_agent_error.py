import sys
import os
sys.path.append(os.path.abspath("smart-cpg-decision-agent"))

import pandas as pd
from src.agent.agent_core import create_cpg_agent

os.environ["OPENAI_API_KEY"] = "fake"

df = pd.DataFrame({'date': ['2022-01-01'], 'sku_id': [101], 'category': ['Beverages'], 'units_sold': [18], 'revenue': [90.0], 'promo_flag': [0], 'price': [5.0], 'inventory_level': [550]})

# Let's see if we can trigger the error with a fake response
from langchain_core.messages import AIMessage
from langchain_core.language_models import FakeMessagesListChatModel
from langchain_core.language_models import FakeListLLM

import src.genai.llm_interface
# ReAct agents typically use string LLMs rather than Chat Models under the hood, but can use both.
src.genai.llm_interface.get_llm = lambda: FakeListLLM(responses=["Thought: Do I need to use a tool? No\nFinal Answer: The oldest date is 2022-01-01."])

agent = create_cpg_agent(df)
try:
    result = agent.invoke({"input": "what are the latest and the oldest date data you have ?"})
    print("RESULT:", result)
except Exception as e:
    import traceback
    traceback.print_exc()
