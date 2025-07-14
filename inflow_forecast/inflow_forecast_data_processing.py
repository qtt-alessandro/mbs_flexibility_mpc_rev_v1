from scipy import stats
import pandas as pd 
from scipy.ndimage import gaussian_filter1d
import numpy as np 

class BoxCoxTransformer:
    def __init__(self):
        self.lambda_ = None
    def transform(self, data):
        
        transformed_data, self.lambda_ = stats.boxcox(data)
        return transformed_data

    def inverse_transform(self, transformed_data):
        if self.lambda_ is None:
            raise ValueError("The transform method must be called before inverse_transform.")
        
        # Inverse transformation
        if self.lambda_ == 0:
            original_data = np.exp(transformed_data)
        else:
            original_data = (np.exp(np.log(self.lambda_ * transformed_data + 1) / self.lambda_))

        return original_data
    
    
def prophet_data_processing(df):
    
    start_date = pd.to_datetime('2023-05-01 00:00:00+00:00')
    date_range = pd.date_range(start=start_date, periods=len(df), freq='H')
    df.index = date_range
    #df["rain"] = gaussian_filter1d(df['precip'].astype(float), sigma=2)
    df = df.reset_index()
    df = df.rename(columns={'index':'ds', 'inflow_kf':'y'})
    df['ds'] = df['ds'].dt.tz_localize(None)
    return df 

def create_prophet_holiday_dataframe(df, feature):
    holiday_df = df[df[feature] > 0][['ds']]
    holiday_df['holiday'] = 'rainy_day'
    holiday_df['lower_window'] = 0
    holiday_df['upper_window'] = 1
    return holiday_df