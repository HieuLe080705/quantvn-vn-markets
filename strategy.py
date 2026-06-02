import pandas as pd
import numpy as np


def gen_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    VN CTA-Style Momentum Breakout Strategy

    position:
        1  = Long
        -1 = Short
        0  = No position
    """

    df = df.copy()

    # =========================
    # Handle column names
    # =========================

    close = df["Close"] if "Close" in df.columns else df["close"]
    high = df["High"] if "High" in df.columns else df["high"]
    low = df["Low"] if "Low" in df.columns else df["low"]
    volume = df["volume"] if "volume" in df.columns else df["Volume"]

    # =========================
    # Indicators
    # =========================

    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    df["EMA200"] = close.ewm(span=200, adjust=False).mean()

    df["High20"] = high.shift(1).rolling(20).max()
    df["Low20"] = low.shift(1).rolling(20).min()

    df["ExitHigh10"] = high.shift(1).rolling(10).max()
    df["ExitLow10"] = low.shift(1).rolling(10).min()

    df["VolumeMA20"] = volume.rolling(20).mean()

    prev_close = close.shift(1)

    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    df["ATR14"] = true_range.rolling(14).mean()
    df["ATR_Pct"] = df["ATR14"] / close

    # =========================
    # Trading conditions
    # =========================

    long_entry = (
        (close > df["High20"]) &
        (close > df["EMA50"]) &
        (df["EMA50"] > df["EMA200"]) &
        (volume > df["VolumeMA20"]) &
        (df["ATR_Pct"] < 0.03)
    )

    short_entry = (
        (close < df["Low20"]) &
        (close < df["EMA50"]) &
        (df["EMA50"] < df["EMA200"]) &
        (volume > df["VolumeMA20"]) &
        (df["ATR_Pct"] < 0.03)
    )

    long_exit = (
        (close < df["ExitLow10"]) |
        (close < df["EMA50"])
    )

    short_exit = (
        (close > df["ExitHigh10"]) |
        (close > df["EMA50"])
    )

    # =========================
    # Generate position
    # =========================

    df["position"] = 0
    position = 0

    for i in range(len(df)):
        if position == 0:
            if long_entry.iloc[i]:
                position = 1
            elif short_entry.iloc[i]:
                position = -1

        elif position == 1:
            if long_exit.iloc[i]:
                position = 0

        elif position == -1:
            if short_exit.iloc[i]:
                position = 0

        df.loc[df.index[i], "position"] = position

    df["position"] = df["position"].fillna(0)

    return df