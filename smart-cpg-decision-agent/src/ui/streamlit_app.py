import streamlit as st
import pandas as pd
import os
import sys

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_loader import DataLoader
from src.agent.agent_core import create_cpg_agent

# Initialize the DataLoader (forcing pandas for local Streamlit use to simplify dependencies)
@st.cache_resource
def load_app_data():
    dl = DataLoader(use_spark=False)
    # Assume the data file is in the root data folder
    data_path = os.path.join(os.path.dirname(__file__), '../../data/cpg_sales_data.parquet')

    # If the file doesn't exist (e.g., this is a fresh clone), create a small dummy df for the UI to load
    if not os.path.exists(data_path):
        st.warning(f"Data file {data_path} not found. Using a dummy dataframe for testing UI.")
        return pd.DataFrame({
            'date': ['2022-01-01'], 'store_id': [1], 'store_region': ['North'],
            'sku_id': [101], 'category': ['Beverages'], 'units_sold': [18],
            'revenue': [90.0], 'promo_flag': [0], 'promo_type': ['None'],
            'price': [5.0], 'inventory_level': [550], 'store_size': ['Medium'],
            'holiday_flag': [0]
        })

    return dl.load_data(data_path)

@st.cache_resource
def get_agent(_df):
    return create_cpg_agent(_df)

def main():
    st.set_page_config(page_title="CPG Decision Support Agent", page_icon="📈", layout="wide")

    st.title("📈 Smart CPG Decision Support Agent")
    st.markdown("Interact with the Agentic AI to analyze multi-store sales data, simulate scenarios, and generate strategy memos.")

    # Load Data
    with st.spinner("Loading synthetic sales data..."):
        df = load_app_data()

    st.sidebar.header("Data Overview")
    st.sidebar.write(f"**Total Records:** {len(df)}")
    st.sidebar.write(f"**Unique Stores:** {df['store_id'].nunique()}")
    st.sidebar.write(f"**Unique SKUs:** {df['sku_id'].nunique()}")
    st.sidebar.write(f"**Categories:** {', '.join(df['category'].unique())}")

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Initialize the LangChain Agent
    try:
        agent = get_agent(df)

        # Accept user input
        if prompt := st.chat_input("Ask the agent to analyze trends or simulate a scenario..."):
            # Display user message in chat message container
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                with st.spinner("Agent is thinking and using tools..."):
                    try:
                        # Run the agent
                        response = agent.run(prompt)
                        st.markdown(response)
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error executing agent: {str(e)}")
                        st.session_state.messages.append({"role": "assistant", "content": f"I encountered an error: {str(e)}"})

    except Exception as e:
        st.error(f"Failed to initialize the Agent. Have you set your LLM API keys in the environment? Error: {str(e)}")
        st.info("Set OPENAI_API_KEY, GOOGLE_API_KEY, or HUGGINGFACEHUB_API_TOKEN in your terminal before running Streamlit.")

if __name__ == "__main__":
    main()
