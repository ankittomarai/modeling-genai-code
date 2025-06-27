import pandas as pd
import numpy as np

def rolling_iqr_outlier_mask(df, date_col, target_col, window_months=12):
    """
    Remove outliers using a rolling 12-month IQR window, strictly using only past data for each row.
    Returns a boolean mask for rows to keep.
    """
    mask = np.ones(len(df), dtype=bool)
    for idx, row in df.iterrows():
        end_date = row[date_col] - pd.Timedelta(days=1)
        start_date = end_date - pd.DateOffset(months=window_months)
        window = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)][target_col]
        if len(window) < 10:
            continue
        Q1 = window.quantile(0.25)
        Q3 = window.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        if not (lower <= row[target_col] <= upper):
            mask[idx] = False
    return mask

def add_rolling_features_12m(df, group_col, target):
    """
    Add 12-month rolling mean, std, and lag1 for target at group_col level, using only past data.
    """
    df = df.sort_values(['Transaction_Date'])
    roll_mean = []
    roll_std = []
    lag1 = []
    for idx, row in df.iterrows():
        end_date = row['Transaction_Date'] - pd.Timedelta(days=1)
        start_date = end_date - pd.DateOffset(months=12)
        window = df[(df[group_col] == row[group_col]) & (df['Transaction_Date'] >= start_date) & (df['Transaction_Date'] <= end_date)][target]
        roll_mean.append(window.mean() if len(window) > 0 else np.nan)
        roll_std.append(window.std() if len(window) > 0 else np.nan)
        lag1.append(window.iloc[-1] if len(window) > 0 else np.nan)
    df[f'{group_col}_margin_rollmean_12m'] = roll_mean
    df[f'{group_col}_margin_rollstd_12m'] = roll_std
    df[f'{group_col}_margin_lag1_12m'] = lag1
    return df

def add_count_features_12m(df, group_col):
    """
    Add 12-month rolling transaction count for group_col, using only past data.
    """
    df = df.sort_values(['Transaction_Date'])
    txn_count = []
    for idx, row in df.iterrows():
        end_date = row['Transaction_Date'] - pd.Timedelta(days=1)
        start_date = end_date - pd.DateOffset(months=12)
        window = df[(df[group_col] == row[group_col]) & (df['Transaction_Date'] >= start_date) & (df['Transaction_Date'] <= end_date)]
        txn_count.append(len(window))
    df[f'{group_col}_txn_count_12m'] = txn_count
    return df

def add_target_encoding_12m(df, group_col):
    """
    Add 12-month rolling mean target encoding for group_col, using only past data.
    """
    df = df.sort_values(['Transaction_Date'])
    target_enc = []
    for idx, row in df.iterrows():
        end_date = row['Transaction_Date'] - pd.Timedelta(days=1)
        start_date = end_date - pd.DateOffset(months=12)
        window = df[(df[group_col] == row[group_col]) & (df['Transaction_Date'] >= start_date) & (df['Transaction_Date'] <= end_date)]['Margin_Per_Unit']
        target_enc.append(window.mean() if len(window) > 0 else np.nan)
    df[f'{group_col}_target_enc_12m'] = target_enc
    return df

def add_segment_mean_6m(df, seg_col, target):
    """
    Add last 6 months mean margin per unit for segment column, using only past data.
    """
    df = df.sort_values(['Transaction_Date'])
    seg_mean_6m = []
    for idx, row in df.iterrows():
        end_date = row['Transaction_Date'] - pd.Timedelta(days=1)
        start_date = end_date - pd.DateOffset(months=6)
        window = df[(df[seg_col] == row[seg_col]) & (df['Transaction_Date'] >= start_date) & (df['Transaction_Date'] <= end_date)][target]
        seg_mean_6m.append(window.mean() if len(window) > 0 else np.nan)
    df[f'{seg_col}_mean_margin_6m'] = seg_mean_6m
    return df

def add_recency_weight(df, date_col='Transaction_Date'):
    """
    Add a recency weight feature: 1 for current month, 0.9 for previous, 0.8 for two months ago, etc.
    """
    latest_month = df[date_col].max().to_period('M')
    recency_weights = []
    for d in df[date_col]:
        months_ago = (latest_month.year - d.year) * 12 + (latest_month.month - d.month)
        weight = max(0, 1 - 0.1 * months_ago)
        recency_weights.append(weight)
    df['Recency_Weight'] = recency_weights
    return df

def engineer_features(df):
    """
    Main feature engineering pipeline. Returns engineered DataFrame and feature list.
    """
    df = df.sort_values('Transaction_Date').reset_index(drop=True)
    # Outlier removal
    outlier_mask = rolling_iqr_outlier_mask(df, 'Transaction_Date', 'Margin_Per_Unit', window_months=12)
    df = df[outlier_mask].reset_index(drop=True)
    # Date features
    df['Year'] = df['Transaction_Date'].dt.year
    df['Month'] = df['Transaction_Date'].dt.month
    df['Quarter'] = df['Transaction_Date'].dt.quarter
    df['DayOfWeek'] = df['Transaction_Date'].dt.dayofweek
    df['Is_End_of_Quarter'] = df['Transaction_Date'].dt.is_quarter_end.astype(int)
    df['Is_End_of_Year'] = df['Transaction_Date'].dt.is_year_end.astype(int)
    df['YearMonth'] = df['Transaction_Date'].dt.to_period('M')
    # Rolling/lag features
    for col in ['Product_Level_2', 'Customer_Name', 'Plant']:
        df = add_rolling_features_12m(df, col, 'Margin_Per_Unit')
    # Interaction features
    df['Customer_Product'] = df['Customer_Name'] + '_' + df['Product_Level_2']
    df['Product_Plant'] = df['Product_Level_2'] + '_' + df['Plant']
    # Segment features
    segment1 = 'segment1_product_plant'
    segment2 = 'segment2_product_application'
    df[segment1] = df['Product_Level_2'] + '_' + df['Plant']
    df[segment2] = df['Product_Level_2'] + '_' + df['Application']
    df = add_segment_mean_6m(df, segment1, 'Margin_Per_Unit')
    df = add_segment_mean_6m(df, segment2, 'Margin_Per_Unit')
    # Frequency/count features
    for col in ['Customer_Name', 'Product_Level_2', 'Plant']:
        df = add_count_features_12m(df, col)
    # Target encoding
    for col in ['Customer_Name', 'Product_Level_2', 'Plant', 'Customer_Product', 'Product_Plant']:
        df = add_target_encoding_12m(df, col)
    # Remove old rolling/count/target enc features (if any)
    old_features = [c for c in df.columns if ('rollmean_' in c and not c.endswith('12m')) or ('rollstd_' in c and not c.endswith('12m')) or ('lag1' in c and not c.endswith('12m')) or ('txn_count_' in c and not c.endswith('12m')) or ('target_enc' in c and not c.endswith('12m'))]
    df = df.drop(columns=old_features)
    # Add recency weight
    df = add_recency_weight(df, 'Transaction_Date')
    # Prepare feature list
    drop_cols = ['Transaction_Date', 'Margin_Per_Unit', 'Total_Value', 'Quantity', 'YearMonth']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    # Fill NA
    df[feature_cols] = df[feature_cols].fillna(-1)
    return df, feature_cols 