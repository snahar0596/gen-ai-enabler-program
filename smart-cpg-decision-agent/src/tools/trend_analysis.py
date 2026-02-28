import pandas as pd
import numpy as np

def calculate_category_trends(df: pd.DataFrame, time_period='W'):
    """
    Calculates sales volume and revenue trends over a specified period (default Weekly 'W').
    This function expects a pandas DataFrame for quick heuristic calculation within tools.
    """
    # Group by the specified period and category
    df['date'] = pd.to_datetime(df['date'])

    # Resample by time_period and category
    trends = df.groupby([pd.Grouper(key='date', freq=time_period), 'category']).agg(
        total_units_sold=('units_sold', 'sum'),
        total_revenue=('revenue', 'sum')
    ).reset_index()

    return trends

def compare_stores_performance(df: pd.DataFrame, metric='revenue'):
    """
    Compares the overall performance of all stores based on a given metric (revenue or units_sold).
    """
    store_performance = df.groupby(['store_id', 'store_region']).agg(
        total_revenue=('revenue', 'sum'),
        total_units_sold=('units_sold', 'sum')
    ).reset_index()

    store_performance = store_performance.sort_values(by=f'total_{metric}', ascending=False)
    return store_performance

def analyze_seasonality(df: pd.DataFrame, category=None):
    """
    Identifies the best selling months for a given category (or overall).
    """
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month

    if category:
        filtered_df = df[df['category'] == category]
    else:
        filtered_df = df

    monthly_sales = filtered_df.groupby('month').agg(
        total_revenue=('revenue', 'sum')
    ).reset_index()

    monthly_sales = monthly_sales.sort_values(by='total_revenue', ascending=False)
    return monthly_sales
