import pandas as pd
import numpy as np

def detect_sales_spikes(df: pd.DataFrame, threshold=2.0):
    """
    Detects abnormal spikes in sales volume where the daily units sold
    exceed the mean units sold + (threshold * standard deviation).
    This function groups by SKU and Store to isolate specific anomalies.
    """
    # Group by SKU and Store to calculate mean and std deviation
    grouped = df.groupby(['sku_id', 'store_id'])

    anomalies = []

    for name, group in grouped:
        mean_sales = group['units_sold'].mean()
        std_sales = group['units_sold'].std()

        # Calculate the threshold limit
        limit = mean_sales + (threshold * std_sales)

        # Find rows where units_sold > limit
        spike_rows = group[group['units_sold'] > limit]

        if not spike_rows.empty:
            anomalies.append(spike_rows)

    if anomalies:
        return pd.concat(anomalies).sort_values(by='date')
    else:
        return pd.DataFrame()

def detect_stock_shortages(df: pd.DataFrame, critical_level=50):
    """
    Detects when a store's inventory for a SKU drops below a critical level.
    """
    shortages = df[df['inventory_level'] < critical_level]
    return shortages.sort_values(by=['date', 'store_id', 'sku_id'])

def flag_anomalous_promotions(df: pd.DataFrame):
    """
    Finds instances where a promotion was active but sales were below average for that SKU/Store,
    indicating a failed promotion.
    """
    # Calculate baseline (non-promo) average per SKU and Store
    baseline = df[df['promo_flag'] == 0].groupby(['sku_id', 'store_id'])['units_sold'].mean().reset_index()
    baseline.rename(columns={'units_sold': 'baseline_avg_units'}, inplace=True)

    # Filter promo days
    promo_days = df[df['promo_flag'] == 1]

    # Merge with baseline
    merged = pd.merge(promo_days, baseline, on=['sku_id', 'store_id'], how='left')

    # Find anomalies: Promo units sold < baseline avg units
    failed_promos = merged[merged['units_sold'] < merged['baseline_avg_units']]

    return failed_promos.sort_values(by='date')
