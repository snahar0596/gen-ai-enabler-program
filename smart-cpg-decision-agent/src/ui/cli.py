import argparse
import sys
import os
import pandas as pd

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_loader import DataLoader
from src.agent.agent_core import create_cpg_agent

def main():
    parser = argparse.ArgumentParser(description="CLI for the Smart CPG Decision Support Agent.")
    parser.add_argument("--data_path", type=str, default="../../data/cpg_sales_data.parquet", help="Path to the synthetic data.")
    args = parser.parse_args()

    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.data_path))

    # Load data using Pandas for CLI
    print("Loading synthetic data...")
    try:
        if not os.path.exists(data_path):
            print(f"Data file {data_path} not found. Creating a dummy dataframe for testing CLI.")
            df = pd.DataFrame({
                'date': ['2022-01-01'], 'store_id': [1], 'store_region': ['North'],
                'sku_id': [101], 'category': ['Beverages'], 'units_sold': [18],
                'revenue': [90.0], 'promo_flag': [0], 'promo_type': ['None'],
                'price': [5.0], 'inventory_level': [550], 'store_size': ['Medium'],
                'holiday_flag': [0]
            })
        else:
            dl = DataLoader(use_spark=False)
            df = dl.load_data(data_path)

        print(f"Data loaded successfully. Total Records: {len(df)}")
    except Exception as e:
        print(f"Failed to load data: {e}")
        sys.exit(1)

    # Initialize Agent
    print("Initializing Agentic AI loop...")
    try:
        agent = create_cpg_agent(df)
        print("Agent initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        print("Please ensure you have set OPENAI_API_KEY, GOOGLE_API_KEY, or HUGGINGFACEHUB_API_TOKEN in your environment.")
        sys.exit(1)

    print("\n" + "="*50)
    print("Welcome to the CPG Decision Support Agent CLI.")
    print("Type 'exit' or 'quit' to terminate the session.")
    print("="*50 + "\n")

    while True:
        try:
            query = input("Ask the agent: ")
            if query.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break

            print("\nThinking...")
            response = agent.run(query)
            print("\nAgent Response:")
            print(response)
            print("\n" + "-"*50 + "\n")

        except KeyboardInterrupt:
            print("\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred while communicating with the agent: {e}")

if __name__ == "__main__":
    main()
