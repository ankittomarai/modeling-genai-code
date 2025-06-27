import numpy as np
import pandas as pd

def apply_pricing_rules(
    df,
    predicted_col='Predicted_Margin_Per_Unit',
    customer_col='Customer_Name',
    product_col='Product_Level_2',
    plant_col='Plant',
    margin_col='Margin_Per_Unit',
    calibration_factor=0.5,
    flag_col='Adjusted_Flag',
    original_col='Original_Prediction',
    min_margin_col='Historical_Min_Margin'
):
    """
    Apply pricing rules:
    1. If predicted margin < historical min for customer-product-plant, adjust upward using calibration_factor.
    2. Add a flag and store the original prediction if modified.
    3. Add the historical min margin as a column for reference.
    """
    # Compute historical min margin for each customer-product-plant
    min_margin = (
        df.groupby([customer_col, product_col, plant_col])[margin_col]
        .min()
        .reset_index()
        .rename(columns={margin_col: min_margin_col})
    )
    # Merge min margin into df
    df = pd.merge(df, min_margin, on=[customer_col, product_col, plant_col], how='left')
    # Apply rules
    adjusted = []
    original = []
    for idx, row in df.iterrows():
        pred = row[predicted_col]
        min_hist = row[min_margin_col]
        if np.isnan(min_hist):
            # No historical margin, do not adjust
            adjusted.append(0)
            original.append(np.nan)
        elif pred < min_hist:
            # Adjust upward
            new_pred = min_hist + calibration_factor * (min_hist - pred)
            df.at[idx, predicted_col] = new_pred
            adjusted.append(1)
            original.append(pred)
        else:
            adjusted.append(0)
            original.append(np.nan)
    df[flag_col] = adjusted
    df[original_col] = original
    return df 