import pandas as pd

def simulate_price_change(df: pd.DataFrame, sku_id: int, price_change_pct: float, elasticity=-1.5):
    """
    Simulates a price hike or decrease on a specific SKU.
    Elasticity determines the volume drop:
    New Volume = Old Volume * (1 + (price_change_pct * elasticity))

    Returns a dataframe of the impacted sales.
    """
    # Filter for the specific SKU
    sku_data = df[df['sku_id'] == sku_id].copy()

    if sku_data.empty:
        return f"No data found for SKU {sku_id}"

    original_revenue = sku_data['revenue'].sum()
    original_volume = sku_data['units_sold'].sum()
    original_price_avg = sku_data['price'].mean()

    # Calculate new price and volume based on elasticity
    new_price_avg = original_price_avg * (1 + price_change_pct)
    volume_change_pct = price_change_pct * elasticity

    # Cap the volume drop to 0
    if volume_change_pct <= -1:
        new_volume = 0
    else:
        new_volume = original_volume * (1 + volume_change_pct)

    new_revenue = new_volume * new_price_avg

    return {
        "sku_id": sku_id,
        "price_change": f"{price_change_pct * 100}%",
        "elasticity_assumption": elasticity,
        "original_avg_price": round(original_price_avg, 2),
        "new_avg_price": round(new_price_avg, 2),
        "original_total_units": int(original_volume),
        "simulated_total_units": int(new_volume),
        "original_revenue": round(original_revenue, 2),
        "simulated_revenue": round(new_revenue, 2),
        "revenue_impact": round(new_revenue - original_revenue, 2)
    }

def simulate_promotion(df: pd.DataFrame, category: str, promo_uplift_pct: float, promo_cost_per_unit: float):
    """
    Simulates running a new promotion across an entire category.
    Assumes a flat percentage uplift in volume and a flat cost per unit sold.
    """
    cat_data = df[df['category'] == category].copy()

    if cat_data.empty:
        return f"No data found for category '{category}'"

    # Baseline calculations
    baseline_volume = cat_data[cat_data['promo_flag'] == 0]['units_sold'].sum()
    baseline_revenue = cat_data[cat_data['promo_flag'] == 0]['revenue'].sum()

    # Simulate new promo volume
    simulated_volume = baseline_volume * (1 + promo_uplift_pct)
    avg_price = cat_data['price'].mean()

    # Calculate revenue and costs
    gross_revenue = simulated_volume * avg_price
    total_promo_cost = simulated_volume * promo_cost_per_unit
    net_revenue = gross_revenue - total_promo_cost

    return {
        "category": category,
        "promo_uplift_assumption": f"{promo_uplift_pct * 100}%",
        "baseline_units": int(baseline_volume),
        "simulated_units": int(simulated_volume),
        "baseline_revenue": round(baseline_revenue, 2),
        "simulated_gross_revenue": round(gross_revenue, 2),
        "total_promo_cost": round(total_promo_cost, 2),
        "simulated_net_revenue": round(net_revenue, 2),
        "net_revenue_impact": round(net_revenue - baseline_revenue, 2)
    }
