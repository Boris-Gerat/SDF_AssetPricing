from Utils import Fred_MD, stock_retrieve
from Utils_Stat import tickers_20

import pandas as pd


def allign_data(df_1: pd.DataFrame, df_2: pd.DataFrame, on_df1: str, on_df2: str):

    df_1[on_df1] = pd.to_datetime(df_1[on_df1], errors="coerce")
    df_2[on_df2] = pd.to_datetime(df_2[on_df2], errors="coerce")

    common_dates = set(df_1[on_df1]).intersection(set(df_2[on_df2]))

    df_1_alligned = df_1[df_1[on_df1].isin(common_dates)].sort_values(on_df1)
    df_2_alligned = df_2[df_2[on_df2].isin(common_dates)].sort_values(on_df2)

    df_1_alligned.reset_index(drop=True)
    df_2_alligned.reset_index(drop=True)

    return df_1_alligned, df_2_alligned


def main():
    factors = Fred_MD("2010-01-01", factors=5, factor_verbose=True)
    stocks = stock_retrieve(tickers=tickers_20, start="2010-01-01")

    #   factors_alligned, stocks_alligned = allign_data(factors, stocks, "sasdate", "Date")

    print(factors.index)


if __name__ == "__main__":
    main()
