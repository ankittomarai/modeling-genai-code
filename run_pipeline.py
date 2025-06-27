import pandas as pd
from feature_engineering import engineer_features
from binning import fit_product_bins, assign_product_bins
from classifier import train_bin_classifier, predict_bin
from regressor import train_regressor, predict_margin
from pricing_rules import apply_pricing_rules
from prediction import generate_pricelist

# 1. Load data
print('Loading data...')
df = pd.read_csv('commodity_sales_data.csv', parse_dates=['Transaction_Date'])

# 2. Feature engineering
print('Running feature engineering...')
df_eng, feature_cols = engineer_features(df)

# 3. Binning: create product-level margin bins and assign bin labels
print('Creating product-level margin bins...')
bin_edges = fit_product_bins(df_eng, n_bins=5)
df_eng = assign_product_bins(df_eng, bin_edges, margin_col='Margin_Per_Unit', product_col='Product_Level_2', label_col='Margin_Bin')

# 4. Train CatBoost classifier to predict margin bins
print('Training CatBoost classifier for margin bins...')
classifier_features = [col for col in feature_cols if col != 'Margin_Bin']  # Exclude label
clf = train_bin_classifier(df_eng, classifier_features, label_col='Margin_Bin')

# 5. Predict bins for all data (for use as feature in regressor)
print('Predicting bins for all data...')
df_eng['Predicted_Bin'] = predict_bin(clf, df_eng, classifier_features)

# 6. Train CatBoost regressor using all features (including predicted bin)
print('Training CatBoost regressor for margin prediction...')
regressor_features = feature_cols + ['Predicted_Bin']
reg = train_regressor(df_eng, regressor_features, target_col='Margin_Per_Unit')

# 7. Prepare latest month test set and all required combinations
latest_month = df_eng['YearMonth'].max()
test_latest = df_eng[df_eng['YearMonth'] == latest_month].copy()
# Ensure all required combos (including outlier-only) are present
all_combos = df[['Customer_Name', 'Product_Level_1', 'Product_Level_2', 'Application', 'Plant']].drop_duplicates().reset_index(drop=True)

# 8. Predict margin for latest month
print('Predicting margin for latest month...')
test_latest['Predicted_Margin_Per_Unit'] = predict_margin(reg, test_latest, regressor_features)

# 9. Apply pricing rules
print('Applying pricing rules...')
test_latest = apply_pricing_rules(test_latest, predicted_col='Predicted_Margin_Per_Unit')

# 10. Generate and save latest pricelist (with price range)
print('Generating latest pricelist...')
pricelist = generate_pricelist(
    test_latest,
    reg,
    regressor_features,
    bin_edges,
    bin_col='Predicted_Bin',
    output_csv='latest_pricelist.csv',
    all_combos_df=all_combos
)
print('Pipeline complete.') 