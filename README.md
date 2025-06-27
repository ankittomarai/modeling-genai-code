# Commodity Market Pricing Pipeline

This project provides a modular, interpretable machine learning pipeline for generating monthly commodity market pricelists, using historical sales data and robust business rules.

## Overview

- **Goal:** Predict margin per unit for each unique customer-product-plant combination, with price ranges and business rule enforcement.
- **Approach:**
  - Product-level margin binning and classification (CatBoost)
  - Regression (CatBoost) with predicted bin as a feature
  - Recency and robust feature engineering
  - Post-processing with pricing rules
  - All code is modular and well-commented for easy understanding

## Stepwise Project Guide

1. **Feature Engineering**

   - Clean and prepare the data.
   - Add rolling statistics, target encoding, and a recency weight (recent transactions are weighted higher).
   - Create interaction and segment features for richer modeling.

2. **Product-Level Margin Binning**

   - For each product, split historical margins into 5 bins (using quantiles or histogram binning).
   - Assign each transaction a bin label based on its margin and product.

3. **Margin Bin Classification**

   - Train a CatBoost classifier to predict the margin bin for each transaction, using all engineered features.
   - The classifier provides a price range (bin) for each quote.

4. **Use Bin as Feature in Regression**

   - Add the predicted bin (from the classifier) as a feature for the next step.

5. **Margin Regression**

   - Train a CatBoost regressor to predict the exact margin per unit, using all features (including the predicted bin).
   - Hyperparameters for both classifier and regressor are tuned using GridSearchCV.

6. **Apply Pricing Rules**

   - If the predicted margin is lower than the historical minimum for a customer-product-plant, adjust it upward using a calibration factor.
   - Add a flag and store the original prediction if an adjustment was made.

7. **Pricelist Generation**
   - Generate the final pricelist, including:
     - Predicted margin
     - Predicted price range (from classifier)
     - Adjustment flag and original value if applicable
     - All required combinations, even those only in outlier records
   - Save the pricelist as `latest_pricelist.csv`.

## Project Structure

```
├── binning.py              # Product-level margin binning functions
├── classifier.py           # CatBoost classifier for margin bins
├── feature_engineering.py  # Feature engineering (recency, rolling, etc.)
├── prediction.py           # Pricelist generation (with price range)
├── pricing_rules.py        # Business/pricing rules enforcement
├── regressor.py            # CatBoost regressor for margin prediction
├── run_pipeline.py         # Main script to run the full pipeline
├── commodity_sales_data.csv# Input data (keep in repo)
├── latest_pricelist.csv    # Output: latest generated pricelist
├── README.md               # This file
├── .gitignore              # Ignores cache, logs, models, but keeps data
```

## How to Run

1. **Install dependencies:**
   - Python 3.8+
   - `pip install pandas numpy scikit-learn catboost`
2. **Place your data:**
   - Ensure `commodity_sales_data.csv` is in the project root.
3. **Run the pipeline:**
   ```bash
   python run_pipeline.py
   ```
4. **Output:**
   - The latest pricelist will be saved as `latest_pricelist.csv`.

## Notes

- All business rules (e.g., minimum margin enforcement, calibration) are in `pricing_rules.py`.
- The pipeline is modular and easy to extend or modify.
- All required combinations (even those only in outlier records) are included in the pricelist.
- The code is heavily commented for clarity and learning.

---

**For any questions or improvements, just open the code and follow the comments!**
