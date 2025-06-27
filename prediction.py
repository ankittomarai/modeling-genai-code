import pandas as pd
import numpy as np
from binning import get_bin_range


def generate_pricelist(
    test_df,
    model,
    feature_cols,
    bin_edges,
    bin_col='Predicted_Bin',
    output_csv='latest_pricelist.csv',
    all_combos_df=None
):
    """
    Generate the latest pricelist using the trained model and test set for the latest month.
    Includes predicted margin, price range (from classifier/binning), and ensures all required combinations are present.
    If all_combos_df is provided, ensures all combinations are included in the output.
    """
    # Get unique combinations for the pricelist
    unique_combos = test_df[['Customer_Name', 'Product_Level_1', 'Product_Level_2', 'Application', 'Plant']].drop_duplicates().reset_index(drop=True)
    if all_combos_df is not None:
        # Union with all required combos (e.g., from outlier records)
        unique_combos = pd.concat([unique_combos, all_combos_df], ignore_index=True).drop_duplicates().reset_index(drop=True)
    # Merge engineered features for these combos
    combo_features = pd.merge(unique_combos, test_df, on=['Customer_Name', 'Product_Level_1', 'Product_Level_2', 'Application', 'Plant'], how='left')
    combo_features = combo_features[feature_cols].fillna(-1)
    # Predict margin
    pred_margin = model.predict(combo_features)
    pricelist = unique_combos.copy()
    pricelist['Predicted_Margin_Per_Unit'] = np.round(pred_margin, 2)
    # Add predicted bin and price range if available
    if bin_col in test_df.columns:
        pricelist[bin_col] = test_df[bin_col].values[:len(pricelist)]
        # Add price range columns
        min_range = []
        max_range = []
        for idx, row in pricelist.iterrows():
            product = row['Product_Level_2']
            bin_label = row.get(bin_col, np.nan)
            rng = get_bin_range(product, int(bin_label) if not pd.isna(bin_label) else 1, bin_edges)
            min_range.append(rng[0])
            max_range.append(rng[1])
        pricelist['Predicted_Min_Range'] = min_range
        pricelist['Predicted_Max_Range'] = max_range
    pricelist.to_csv(output_csv, index=False)
    print(f"Pricelist saved as {output_csv} with {len(pricelist)} rows.")
    return pricelist 