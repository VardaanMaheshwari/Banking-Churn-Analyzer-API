"""
Bank Customer Churn — Feature Engineering & EDA helpers.

Mirrors the helper-driven style of the reference Telco repo
(grab_col_names / check_data / outlier & missing-value utilities) but adapts
every step to the Bank Churn dataset (Churn_Modelling.csv), whose columns are
completely different from the Telco set.

The notebook imports from this module so the analysis stays DRY and the same
transforms are reused at training and (indirectly) at serving time.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. INSPECTION HELPERS  (repo-style)
# ---------------------------------------------------------------------------
def check_data(dataframe: pd.DataFrame, head: int = 5) -> None:
    """One-shot health check: shape, dtypes, head/tail, NaNs, describe."""
    print(20 * "-" + " SHAPE " + 20 * "-")
    print(dataframe.shape)
    print(20 * "-" + " DTYPES " + 20 * "-")
    print(dataframe.dtypes)
    print(20 * "-" + " HEAD " + 20 * "-")
    print(dataframe.head(head))
    print(20 * "-" + " MISSING " + 20 * "-")
    print(dataframe.isnull().sum())
    print(20 * "-" + " DESCRIBE " + 20 * "-")
    print(dataframe.describe([0.01, 0.25, 0.50, 0.75, 0.99]).T)


def grab_col_names(dataframe: pd.DataFrame, cat_th: int = 10, car_th: int = 20):
    """Split columns into categorical / numeric / cardinal groups.

    Numeric-looking columns with few distinct values (e.g. HasCrCard) are
    treated as categorical; high-cardinality text columns (e.g. Surname) are
    flagged as cardinal so they can be dropped rather than encoded.
    """
    cat_cols = [c for c in dataframe.columns if dataframe[c].dtype == "O"]
    num_but_cat = [c for c in dataframe.columns
                   if dataframe[c].nunique() < cat_th and dataframe[c].dtype != "O"]
    cat_but_car = [c for c in dataframe.columns
                   if dataframe[c].nunique() > car_th and dataframe[c].dtype == "O"]
    cat_cols = [c for c in cat_cols + num_but_cat if c not in cat_but_car]
    num_cols = [c for c in dataframe.columns
                if dataframe[c].dtype != "O" and c not in num_but_cat]

    print(f"Observations : {dataframe.shape[0]}")
    print(f"Variables    : {dataframe.shape[1]}")
    print(f"cat_cols     : {len(cat_cols)} -> {cat_cols}")
    print(f"num_cols     : {len(num_cols)} -> {num_cols}")
    print(f"cat_but_car  : {len(cat_but_car)} -> {cat_but_car}")
    return cat_cols, num_cols, cat_but_car


def missing_values_table(dataframe: pd.DataFrame, na_name: bool = False):
    """Print a count/ratio table of missing values."""
    na_cols = [c for c in dataframe.columns if dataframe[c].isnull().sum() > 0]
    n_miss = dataframe[na_cols].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_cols].isnull().sum() / len(dataframe) * 100).round(2)
    table = pd.concat([n_miss, ratio], axis=1, keys=["n_miss", "ratio"])
    print(table if len(na_cols) else "No missing values.")
    if na_name:
        return na_cols


# ---------------------------------------------------------------------------
# 2. OUTLIER HELPERS  (IQR method, repo-style winsorization)
# ---------------------------------------------------------------------------
def outlier_thresholds(dataframe, col, q1=0.05, q3=0.95):
    """IQR-based limits. Wider quantiles (5/95) than the textbook 25/75 so we
    only clip genuinely extreme values, not the natural spread of the data."""
    quartile1 = dataframe[col].quantile(q1)
    quartile3 = dataframe[col].quantile(q3)
    iqr = quartile3 - quartile1
    return quartile1 - 1.5 * iqr, quartile3 + 1.5 * iqr


def check_outlier(dataframe, col) -> bool:
    low, up = outlier_thresholds(dataframe, col)
    return bool(dataframe[(dataframe[col] < low) | (dataframe[col] > up)].shape[0] > 0)


def replace_with_thresholds(dataframe, col) -> None:
    """Winsorize: cap values outside the IQR fence to the fence.

    Full-column reassignment (not partial .loc) so an int column can safely
    take float fence values under pandas' strict dtype rules.
    """
    low, up = outlier_thresholds(dataframe, col)
    dataframe[col] = dataframe[col].clip(lower=low, upper=up)


# ---------------------------------------------------------------------------
# 3. CLEANING + FEATURE ENGINEERING  (bank-specific)
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifiers, remove duplicates, standardise categorical text."""
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
    df = df.drop_duplicates()
    for col in ["Geography", "Gender"]:
        df[col] = df[col].astype(str).str.strip().str.title()
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain features for the bank dataset, each with a stated hypothesis.

    (The Telco repo's gender x service interactions don't exist here, so these
    are the bank-appropriate equivalents.)
    """
    # H1: a zero balance is a distinct behavioural state, not just a low number
    df["ZeroBalance"] = (df["Balance"] == 0).astype(int)
    # H2: balance relative to income captures engagement better than raw balance
    df["BalanceToSalary"] = (df["Balance"] / df["EstimatedSalary"]).round(4)
    # H3: product uptake per year of tenure = adoption speed, not just total held
    df["ProductsPerTenure"] = (df["NumOfProducts"] / (df["Tenure"] + 1)).round(4)
    # H4: engagement composite — card AND active differs from either alone
    df["EngagementScore"] = df["IsActiveMember"] + df["HasCrCard"]
    # H5: life stage — banking needs shift by age bracket
    df["AgeBucket"] = pd.cut(df["Age"], bins=[17, 30, 40, 50, 60, 100],
                             labels=["18-30", "31-40", "41-50", "51-60", "60+"])
    return df


NUM_COLS_TO_SCALE = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary",
                     "BalanceToSalary", "ProductsPerTenure"]


def build_model_frame(raw: pd.DataFrame):
    """Full pipeline: clean -> engineer -> encode. Returns (X, y, feature_names).

    One-hot encodes Geography/Gender (drop_first) and drops the AgeBucket
    display column (its information already lives in Age).
    """
    df = clean(raw.copy())
    df = add_features(df)
    df = df.drop(columns=["AgeBucket"])
    df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)
    # keep dummy columns as ints (0/1) rather than bools for clean model input
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
    y = df["Exited"]
    X = df.drop(columns=["Exited"])
    return X, y, list(X.columns)
