from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
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

    # -------------------------
    # Wrap Python functions as Tools
    # -------------------------
    tools = [
        Tool(
            name="CategoryTrends",
            func=lambda period: calculate_category_trends(df, time_period=period).to_string(),
            description="Get sales and revenue trends over a period (e.g., 'W' weekly, 'M' monthly). Input: period string."
        ),
        Tool(
            name="StorePerformance",
            func=lambda metric: compare_stores_performance(df, metric=metric).to_string(),
            description="Compare overall store performance. Input: 'revenue' or 'units_sold'."
        ),
        Tool(
            name="SeasonalityAnalysis",
            func=lambda category: analyze_seasonality(
                df, category=category if category != "all" else None
            ).to_string(),
            description="Identify best-selling months. Input: category name or 'all'."
        ),
        Tool(
            name="SalesSpikes",
            func=lambda threshold: detect_sales_spikes(
                df, threshold=float(threshold)
            ).to_string(),
            description="Find abnormal sales spikes. Input: threshold multiplier (e.g., 2.0)."
        ),
        Tool(
            name="StockShortages",
            func=lambda level: detect_stock_shortages(
                df, critical_level=int(level)
            ).to_string(),
            description="Find stores with low stock. Input: inventory level (e.g., 50)."
        ),
        Tool(
            name="FailedPromotions",
            func=lambda _: flag_anomalous_promotions(df).to_string(),
            description="Find promotions that underperformed. Input ignored."
        ),
        Tool(
            name="SimulatePriceChange",
            func=lambda args: simulate_price_change(
                df,
                sku_id=int(args.split(",")[0]),
                price_change_pct=float(args.split(",")[1]),
                elasticity=float(args.split(",")[2]) if len(args.split(",")) > 2 else -1.5
            ),
            description="Simulate price change. Input: 'sku_id,price_change_pct,elasticity'"
        ),
        Tool(
            name="SimulatePromotion",
            func=lambda args: simulate_promotion(
                df,
                category=args.split(",")[0],
                promo_uplift_pct=float(args.split(",")[1]),
                promo_cost_per_unit=float(args.split(",")[2])
            ),
            description="Simulate promotion. Input: 'category,promo_uplift_pct,promo_cost_per_unit'"
        ),
    ]

    # -------------------------
    # System Prompt
    # -------------------------
    system_message = """You are an expert Decision Support Agent for a Consumer Packaged Goods (CPG) company.
Your goal is to help business heads understand sales data, detect anomalies, and simulate business scenarios to generate actionable strategy memos.
Use the tools provided to answer the user's questions based on the synthetic data.
Always summarize findings clearly and concisely.
If simulating a scenario, provide a brief interpretation of financial impact."""

    # -------------------------
    # Build Prompt (Required in v0.1+)
    # -------------------------
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # -------------------------
    # Create Agent (NEW WAY)
    # -------------------------
    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    # -------------------------
    # Wrap in Executor
    # -------------------------
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
    )

    return agent_executor


def run_agent(agent, query: str):
    """
    Runs a query through the initialized agent.
    """
    return agent.invoke({"input": query})
