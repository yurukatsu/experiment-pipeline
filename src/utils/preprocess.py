from typing import Any

import polars as pl

TRUTHY = {"true", "t", "1", "y", "yes", "真", "True", "TRUE", "１", "はい"}
FALSY = {"false", "f", "0", "n", "no", "偽", "False", "FALSE", "０", "いいえ"}


def _coerce_booleans(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert Utf8 boolean representation columns to Boolean (supports Japanese 真/偽)

    :param df: The input DataFrame.
    :type df: pl.DataFrame
    :return: The DataFrame with boolean columns.
    :rtype: pl.DataFrame
    """
    out = df.clone()
    for name, dtype in df.schema.items():
        if dtype == pl.Utf8:
            # Sample judgment
            s = df[name].drop_nulls()
            if s.is_empty():
                continue
            lower = s.str.to_lowercase()
            rate = lower.is_in(list(TRUTHY)).mean() + lower.is_in(list(FALSY)).mean()
            if rate >= 0.8:  # 8割以上が真偽語ならブール化
                out = out.with_columns(
                    pl.when(pl.col(name).str.to_lowercase().is_in(list(TRUTHY)))
                    .then(True)
                    .when(pl.col(name).str.to_lowercase().is_in(list(FALSY)))
                    .then(False)
                    .otherwise(None)
                    .alias(name)
                    .cast(pl.Boolean)
                )
    return out


def _select_cols(
    df: pl.DataFrame, target: str
) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Select columns from the DataFrame based on their data types.

    :param df: The input DataFrame.
    :type df: pl.DataFrame
    :param target: The target column.
    :type target: str
    :return: A tuple containing the numeric, boolean, categorical, and date columns.
    :rtype: tuple[list[str], list[str], list[str], list[str]]
    """
    num_cols = [
        c
        for c, t in df.schema.items()
        if t
        in (
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
            pl.Float32,
            pl.Float64,
        )
        and c != target
    ]
    bool_cols = [c for c, t in df.schema.items() if t == pl.Boolean]
    cat_cols = [c for c, t in df.schema.items() if t in (pl.Utf8, pl.Categorical)]
    date_cols = [c for c, t in df.schema.items() if t in (pl.Date, pl.Datetime)]
    return num_cols, bool_cols, cat_cols, date_cols


def _impute_scale_numeric(df: pl.DataFrame, cols: list[str]):
    """
    Impute missing values with the median and standardize the columns.

    :param df: The input DataFrame.
    :type df: pl.DataFrame
    :param cols: The columns to impute and scale.
    :type cols: list[str]
    :return: The scaled DataFrame and the statistics used for scaling.
    :rtype: tuple[pl.DataFrame, dict[str, Any]]
    """
    if not cols:
        return df, {}
    stats = df.select(
        [pl.col(c).median().alias(f"{c}__med") for c in cols]
        + [pl.col(c).mean().alias(f"{c}__mean") for c in cols]
        + [pl.col(c).std().alias(f"{c}__std") for c in cols]
    ).to_dicts()[0]
    exprs = []
    for c in cols:
        med = stats[f"{c}__med"]
        mu = stats[f"{c}__mean"]
        std = stats[f"{c}__std"] or 1.0
        exprs += [
            pl.col(c).fill_null(med).alias(c),
            ((pl.col(c).fill_null(med) - mu) / std).alias(f"{c}__z"),
        ]
    scaled = df.with_columns(exprs)
    return scaled, stats


def _impute_booleans(df: pl.DataFrame, cols: list[str]):
    """
    Impute missing boolean values with False and create derived columns for integer conversion.

    :param df: The input DataFrame.
    :type df: pl.DataFrame
    :param cols: The columns to impute.
    :type cols: list[str]
    :return: The DataFrame with imputed boolean columns.
    :rtype: pl.DataFrame
    """
    if not cols:
        return df
    return df.with_columns(
        *[pl.col(c).fill_null(False).alias(c) for c in cols],
        *[(pl.col(c).fill_null(False).cast(pl.Int8).alias(f"{c}__i")) for c in cols],
    )


def _encode_categoricals(df: pl.DataFrame, cols: list[str], onehot_max_card: int = 200):
    """
    Encode categorical features.

    Case 1: Low to medium cardinality: One-Hot Encoding
    Case 2: High cardinality: 64-bit hash (stable) compression

    :param df: The input DataFrame.
    :type df: pl.DataFrame
    :param cols: The columns to encode.
    :type cols: list[str]
    :param onehot_max_card: The maximum cardinality for one-hot encoding.
    :type onehot_max_card: int
    :return: The encoded DataFrame and the artifacts.
    :rtype: tuple[pl.DataFrame, dict[str, list[str]]]
    """
    if not cols:
        return df, {"onehot_cols": [], "hash_cols": [], "levels": {}}

    artifacts = {"onehot_cols": [], "hash_cols": [], "levels": {}}
    out = df

    for c in cols:
        nunique = int(df.select(pl.col(c).n_unique()).item())
        col_nonnull = pl.col(c).fill_null("Missing")

        if nunique <= onehot_max_card:
            levels = df.select(col_nonnull.unique().sort()).to_series().to_list()
            artifacts["levels"][c] = levels
            artifacts["onehot_cols"].append(c)
            out = out.with_columns(col_nonnull.alias(c))
            out = out.to_dummies(columns=[c])
        else:
            artifacts["hash_cols"].append(c)
            out = out.with_columns(
                pl.col(c)
                .fill_null("Missing")
                .hash(seed=0)
                .alias(f"{c}__hash")
                .cast(pl.Int64)
            )
    return out, artifacts


def preprocess(
    df: pl.DataFrame,
    target: str,
    derive_date_features: bool = False,
    onehot_max_card: int = 200,
) -> tuple[pl.DataFrame, Any]:
    assert target in df.columns, f"target '{target}' not found"
    y = df.select(pl.col(target))
    X = df.drop([target])

    # 1. Coerce booleans
    X = _coerce_booleans(X)

    # 2. Select columns
    num_cols, bool_cols, cat_cols, date_cols = _select_cols(X, target=None)

    # 3. Derive date features if needed
    if derive_date_features:
        if date_cols:
            derived = []
            for c in date_cols:
                if X.schema[c] == pl.Date:
                    derived += [
                        pl.col(c).dt.weekday().alias(f"{c}__wday"),
                        pl.col(c).dt.month().alias(f"{c}__month"),
                        pl.col(c).dt.year().alias(f"{c}__year"),
                    ]
                else:
                    derived += [
                        pl.col(c).dt.weekday().alias(f"{c}__wday"),
                        pl.col(c).dt.month().alias(f"{c}__month"),
                        pl.col(c).dt.year().alias(f"{c}__year"),
                    ]
            X = X.with_columns(derived)

            for c in list(X.columns):
                if (
                    c.endswith("__wday")
                    or c.endswith("__month")
                    or c.endswith("__year")
                ):
                    num_cols.append(c)

    # 4. Impute numeric columns and standardize
    X, num_stats = _impute_scale_numeric(X, num_cols)

    # 5. Impute boolean columns
    X = _impute_booleans(X, bool_cols)

    # 6. Impute categorical columns
    if cat_cols:
        X = X.with_columns(*[pl.col(c).fill_null("Missing").alias(c) for c in cat_cols])

    # 7, One-Hot or Hash Encoding
    if cat_cols:
        X, cat_art = _encode_categoricals(X, cat_cols, onehot_max_card=onehot_max_card)

    #  8. Select final columns
    # - Numeric: Select "__z" columns only (keep original values if needed)
    keep_num_z = [f"{c}__z" for c in num_cols]
    keep_bool_i = [f"{c}__i" for c in bool_cols if f"{c}__i" in X.columns]
    keep_hash = [
        f"{c}__hash" for c in cat_art["hash_cols"] if f"{c}__hash" in X.columns
    ]
    # - Categorical: Drop original columns if not needed
    # "col_level" is increased for OHE columns. Drop categorical columns if not needed
    drop_raw = [c for c in cat_cols if c in X.columns]

    selected_cols = [
        c
        for c in X.columns
        if c in keep_num_z or c in keep_bool_i or c in keep_hash or (c not in drop_raw)
    ]
    X = X.select(selected_cols)

    artifacts = {
        "numeric_stats": num_stats,  # {col__med, col__mean, col__std}
        "onehot_levels": cat_art["levels"],
        "onehot_cols": cat_art["onehot_cols"],
        "hash_cols": cat_art["hash_cols"],
        "date_cols": date_cols,
        "bool_cols": bool_cols,
        "num_cols": num_cols,
        "target": target,
        "onehot_max_card": onehot_max_card,
    }

    return X, y, artifacts
