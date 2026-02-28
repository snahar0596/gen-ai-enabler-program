import pytest
import pandas as pd
from src.tools.scenario_simulation import simulate_price_change

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'sku_id': [101, 101, 102],
        'revenue': [100, 200, 50],
        'units_sold': [10, 20, 5],
        'price': [10.0, 10.0, 10.0]
    })

def test_simulate_price_change(sample_data):
    result = simulate_price_change(sample_data, sku_id=101, price_change_pct=0.1, elasticity=-1.0)

    # Assert return type and valid keys
    assert isinstance(result, dict)
    assert result['sku_id'] == 101

    # Original volume is 30. Price goes up 10%. Elasticity is -1.0. Volume should drop 10%. New volume = 27
    assert result['original_total_units'] == 30
    assert result['simulated_total_units'] == 27

    # Original revenue is 300. New price is 11. New volume 27. New revenue = 297
    assert result['original_revenue'] == 300.0
    assert result['simulated_revenue'] == 297.0
