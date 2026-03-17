import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import yfinance as yf
from Utils_Stat import tickers_100, tickers_20


def apply_transformation(series, code):
    match int(float(code)):
        case 1:
            return series
        case 2:
            return series.diff()
        case 3:
            return series.diff().diff()
        case 4:
            return np.log(series)
        case 5:
            return np.log(series).diff()
        case 6:
            return np.log(series).diff().diff()
        case _:
            return series


def Fred_MD(
    start: str | None = None,
    end: str | None = None,
    factors: int | None = None,
    factor_verbose: bool = False,
    quarterly: bool = False,
) -> pd.DataFrame:
    link = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/2025-12-md.csv?sc_lang=en&hash=14BCC7AA1D5AB89D3459B69B8AE67D10"
    data_raw = pd.read_csv(link)
    transform = data_raw.iloc[0]
    transform = transform.drop(labels=["sasdate"], errors="ignore")
    data = data_raw.iloc[1:].copy()
    data["sasdate"] = pd.to_datetime(data["sasdate"])
    data = data.set_index("sasdate").sort_index()
    data = data.apply(pd.to_numeric, errors="coerce")

    # Apply transformations to full history so diffs are computed correctly
    for col in data.columns:
        code = transform[col]
        data[col] = apply_transformation(data[col], code)

    # Date filter AFTER transformations
    if start is not None:
        data = data.loc[pd.to_datetime(start) :]
    if end is not None:
        data = data.loc[: pd.to_datetime(end)]

    # Drop columns that are more than 20% missing in the requested window
    # — these are series that simply don't exist for your sample period
    col_threshold = int(0.8 * len(data))
    data = data.dropna(axis=1, thresh=col_threshold)

    # Forward/backward fill remaining interior NAs (tiny gaps, not structural)
    # then drop any row still fully empty
    data = data.ffill().bfill()
    data = data.dropna(how="any")

    if factor_verbose:
        print(f"Shape after cleaning : {data.shape}")
        print(
            f"Date range           : {data.index[0].date()} to {data.index[-1].date()}"
        )

    # Aggregate to quarterly BEFORE PCA if requested
    if quarterly:
        data = data.resample("QE").mean()

    if factors is not None:
        X = (data - data.mean()) / data.std()
        pca = PCA(n_components=factors)
        F = pca.fit_transform(X)
        factors_df = pd.DataFrame(
            F, index=X.index, columns=[f"PC{i+1}" for i in range(factors)]
        )
        if factor_verbose:
            for i, r in enumerate(pca.explained_variance_ratio_):
                print(f"PC{i+1} explained variance : {r:.1%}")
        return factors_df

    return data


def stock_retrieve(
    tickers: list, start: str, end: str | None = None, returns: bool = False
):
    start = start.strip()
    stock_raw = yf.download(start=start, tickers=tickers, interval="1mo")
    stock_close = stock_raw["Close"]
    if returns == True:
        stock_close = stock_close.pct_change()

    return stock_close


def main():

    stock_data = stock_retrieve(tickers_20, start="2020-01-01")
    print(stock_data)


if __name__ == "__main__":
    main()
