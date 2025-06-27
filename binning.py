import pandas as pd
import numpy as np

def fit_product_bins(df, n_bins=5, margin_col='Margin_Per_Unit', product_col='Product_Level_2'):
    """
    Compute bin edges for each product using historical margin data.
    Returns a dict: {product: bin_edges}
    """
    bin_edges = {}
    for product, group in df.groupby(product_col):
        # Use quantile-based binning (qcut) for equal-sized bins
        try:
            edges = pd.qcut(group[margin_col], q=n_bins, retbins=True, duplicates='drop')[1]
        except ValueError:
            # If not enough unique values, use min/max
            edges = np.linspace(group[margin_col].min(), group[margin_col].max(), n_bins+1)
        bin_edges[product] = edges
    return bin_edges

def assign_product_bins(df, bin_edges, margin_col='Margin_Per_Unit', product_col='Product_Level_2', label_col='Margin_Bin'):
    """
    Assign a bin label (1-n_bins) to each row based on product and margin.
    Adds a new column with bin label.
    """
    labels = []
    for idx, row in df.iterrows():
        product = row[product_col]
        margin = row[margin_col]
        edges = bin_edges.get(product)
        if edges is not None:
            # Bin label is 1-based (1 to n_bins)
            bin_label = np.digitize(margin, edges, right=True)
            # Clamp to 1-n_bins
            bin_label = max(1, min(len(edges)-1, bin_label))
        else:
            bin_label = np.nan
        labels.append(bin_label)
    df[label_col] = labels
    return df

def get_bin_range(product, bin_label, bin_edges):
    """
    Get the price range (min, max) for a given product and bin label.
    """
    edges = bin_edges.get(product)
    if edges is None or bin_label < 1 or bin_label > len(edges)-1:
        return (np.nan, np.nan)
    return (edges[bin_label-1], edges[bin_label]) 