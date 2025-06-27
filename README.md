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
