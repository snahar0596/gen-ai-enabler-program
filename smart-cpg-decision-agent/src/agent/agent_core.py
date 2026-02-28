from langchain.agents.agent_types import AgentType
from langchain.tools import Tool
from langchain.agents.initialize import initialize_agent
from typing import List
import pandas as pd

# Import tools
from src.tools.trend_analysis import calculate_category_trends, compare_stores_performance, analyze_seasonality
from src.tools.anomaly_detection import detect_sales_spikes, detect_stock_shortages, flag_anomalous_promotions
from src.tools.scenario_simulation import simulate_price_change, simulate_promotion
from src.genai.llm_interface import get_llm
from src.agent.memory import get_memory

def create_cpg_agent(df: pd.DataFrame):
    """
    Creates the LangChain Agent loop by binding the tools and the LLM.
    """
    llm = get_llm()
    memory = get_memory()

    # Wrap the python functions as LangChain Tools
    tools = [
        Tool(
            name="CategoryTrends",
            func=lambda period: calculate_category_trends(df, time_period=period).to_string(),
            description="Use this to get sales and revenue trends over a period (e.g., 'W' for weekly, 'M' for monthly). Input the period string."
        ),
        Tool(
            name="StorePerformance",
            func=lambda metric: compare_stores_performance(df, metric=metric).to_string(),
            description="Use this to compare overall performance of all stores. Input the metric ('revenue' or 'units_sold')."
        ),
        Tool(
            name="SeasonalityAnalysis",
            func=lambda category: analyze_seasonality(df, category=category if category != 'all' else None).to_string(),
            description="Use this to identify best selling months for a category. Input the category name or 'all'."
        ),
        Tool(
            name="SalesSpikes",
            func=lambda threshold: detect_sales_spikes(df, threshold=float(threshold)).to_string(),
            description="Use this to find abnormal sales spikes. Input the threshold multiplier (e.g., 2.0)."
        ),
        Tool(
            name="StockShortages",
            func=lambda level: detect_stock_shortages(df, critical_level=int(level)).to_string(),
            description="Use this to find stores with stock below a critical level. Input the inventory level (e.g., 50)."
        ),
        Tool(
            name="FailedPromotions",
            func=lambda _: flag_anomalous_promotions(df).to_string(),
            description="Use this to find promotions that performed worse than average non-promo days. Input is ignored."
        ),
        Tool(
            name="SimulatePriceChange",
            func=lambda args: simulate_price_change(
                df,
                sku_id=int(args.split(",")[0]),
                price_change_pct=float(args.split(",")[1]),
                elasticity=float(args.split(",")[2]) if len(args.split(",")) > 2 else -1.5
            ),
            description="Simulate a price change for a SKU. Input a comma-separated string: 'sku_id,price_change_pct,elasticity' (e.g., '101,0.1,-1.5' for a 10% hike on sku 101)."
        ),
        Tool(
            name="SimulatePromotion",
            func=lambda args: simulate_promotion(
                df,
                category=args.split(",")[0],
                promo_uplift_pct=float(args.split(",")[1]),
                promo_cost_per_unit=float(args.split(",")[2])
            ),
            description="Simulate a promotion for a category. Input a comma-separated string: 'category,promo_uplift_pct,promo_cost_per_unit' (e.g., 'Beverages,0.2,1.5' for a 20% uplift with $1.5 cost)."
        )
    ]

    system_message = """You are an expert Decision Support Agent for a Consumer Packaged Goods (CPG) company.
Your goal is to help business heads understand sales data, detect anomalies, and simulate business scenarios to generate actionable strategy memos.
Use the tools provided to answer the user's questions based on the synthetic data. Always summarize your findings clearly and concisely.
If simulating a scenario, provide a brief interpretation of the financial impact."""

    # Initialize the agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        agent_kwargs={
            "system_message": system_message
        }
    )

    return agent

def run_agent(agent, query: str):
    """
    Runs a query through the initialized agent.
    """
    return agent.run(query)
