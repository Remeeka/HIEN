"""
Reusable demand forecasting module for backend integration.
"""
import pandas as pd
import numpy as np
import joblib

class DemandForecaster:
    def __init__(self, model_path='/content/model.joblib', encoder_path='/content/encoder.joblib'):
        self.model = joblib.load(model_path)
        self.encoder = joblib.load(encoder_path)
        self.cat_cols = ['day_of_week', 'hub_id', 'hub_type', 'city', 'product_id', 'product_category', 'festival', 'season']
        self.num_cols_to_impute = [
            'product_price', 'historical_demand_14d', 'average_daily_demand_30d',
            'current_stock', 'stockout_days_last_30d', 'incoming_stock',
            'competitor_price_index', 'hub_current_utilization',
            'nearby_customer_density', 'average_delivery_time_hours'
        ]

    def preprocess_and_feature_engineer(self, df):
        """Processes raw input data to match training feature space."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'], dayfirst=True)
        
        # 1. Sort chronologically
        df = df.sort_values(by=['hub_id', 'product_id', 'date']).reset_index(drop=True)

        # 2. Impute missing values with pre-defined strategies/medians
        df['festival'] = df['festival'].fillna('No_Festival')
        df['local_event_flag'] = df['local_event_flag'].fillna(0)
        df['discount_percentage'] = df['discount_percentage'].fillna(0.0)

        for col in self.num_cols_to_impute:
            # Imputing using training proxy (for production, real-time fallback could be used)
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)

        # 3. Create Lags & Rolling Features
        grouped = df.groupby(['hub_id', 'product_id'])
        df['demand_7d_lag_1'] = grouped['historical_demand_7d'].shift(1).fillna(0)
        df['demand_7d_lag_2'] = grouped['historical_demand_7d'].shift(2).fillna(0)
        
        df['demand_7d_roll_mean_3'] = grouped['historical_demand_7d'].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        ).fillna(0)
        
        df['demand_7d_roll_std_3'] = grouped['historical_demand_7d'].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).std()
        ).fillna(0)

        # 4. Ordinal Encode Categorical Variables
        df[self.cat_cols] = self.encoder.transform(df[self.cat_cols].astype(str))

        return df

    def predict(self, raw_df):
        """Preprocess and generate non-negative demand predictions."""
        preprocessed_df = self.preprocess_and_feature_engineer(raw_df)
        
        # Define model features
        ignore_cols = ['date', 'future_demand_7d']
        feature_cols = [col for col in preprocessed_df.columns if col not in ignore_cols]
        
        X = preprocessed_df[feature_cols]
        preds = self.model.predict(X)
        
        # Clip negative predictions to 0
        preprocessed_df['predicted_demand_7d'] = np.clip(preds, 0, None)
        return preprocessed_df

if __name__ == "__main__":
    # Example usage test
    print("Forecaster class initialized successfully.")
