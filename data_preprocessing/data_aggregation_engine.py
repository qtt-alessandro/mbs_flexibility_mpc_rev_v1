import requests
from datetime import datetime
from entsoe import EntsoePandasClient
import pandas as pd

def fetch_CO2_data(start_date: str, end_date: str, price_area: str, resample: bool):
    base_url = "https://api.energidataservice.dk/dataset/CO2EmisProg"
    params = {
        "offset": 0,
        "start": f"{start_date}T00:00",
        "end": f"{end_date}T00:00",
        "filter": f'{{"PriceArea":["{price_area}"]}}',
        "sort": "Minutes5UTC ASC"
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        co2_data = pd.DataFrame(data["records"])
        co2_data["Minutes5UTC"] = pd.to_datetime(co2_data["Minutes5UTC"])
        co2_data = co2_data.set_index("Minutes5UTC")
        co2_data = co2_data.drop(columns=["PriceArea", "Minutes5DK"])
        co2_data.index = co2_data.index.rename('time')
        if resample:
            co2_data = co2_data.resample("1h").sum()
            co2_data.index = co2_data.index.tz_localize('UTC')
        return co2_data
    else:
        response.raise_for_status()

def fetch_day_ahead_prices(start_date: str, end_date: str, country_code: str):
    
    key = ''
    client = EntsoePandasClient(api_key=key)
    start = pd.Timestamp(start_date.replace("-", ""), tz='Europe/Copenhagen')
    end = pd.Timestamp(end_date.replace("-", ""), tz='Europe/Copenhagen')
    
    day_ahead_prices_df = client.query_day_ahead_prices(country_code, start=start, end=end)
    day_ahead_prices_df.index = day_ahead_prices_df.index.tz_convert('UTC')
    day_ahead_prices_df.index = day_ahead_prices_df.index.rename("time")
    day_ahead_prices_df = pd.DataFrame(day_ahead_prices_df, columns=["price"])
    return day_ahead_prices_df