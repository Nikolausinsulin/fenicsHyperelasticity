import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt


def interpolate_dataframe(df, x_values):
    """
    Linearly interpolate y-values for arbitrary x-values.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing columns 'x' and 'y'.
    x_values : array-like
        x-values at which to interpolate.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns 'x' and 'y', where 'y' contains the
        interpolated values.
    """

    xColumn = df.columns[0]
    yColumn = df.columns[1]

    # Ensure data are sorted by x
    df_sorted = df.sort_values(xColumn)

    x = df_sorted[xColumn].to_numpy()
    y = df_sorted[yColumn].to_numpy()

    x_values = np.asarray(x_values)

    # Linear interpolation
    y_interp = np.interp(x_values, x, y)

    return pd.DataFrame({xColumn: x_values, yColumn: y_interp})


def fittingError(dfGroundTruth, dfSimulated):
    minStrainContainedInBothDatasets = np.max(
        [np.min(dfGroundTruth["nominalStrain"]), np.min(dfSimulated["nominalStrain"])]
    )
    maxStrainContainedInBothDatasets = np.min(
        [np.max(dfGroundTruth["nominalStrain"]), np.max(dfSimulated["nominalStrain"])]
    )
    xArray = np.linspace(
        minStrainContainedInBothDatasets, maxStrainContainedInBothDatasets, 1000
    )  # just interpolate to high number of points to approximate integral via sum.
    dfGroundTruthInterpolated = interpolate_dataframe(dfGroundTruth, xArray)
    dfSimulatedInterpolated = interpolate_dataframe(dfSimulated, xArray)

    df = dfGroundTruthInterpolated
    df.rename(
        columns={df.columns[0]: "nominalStrain", df.columns[1]: "stressGroundTruth"},
        inplace=True,
    )
    df["stressSimulated"] = dfSimulatedInterpolated[dfSimulatedInterpolated.columns[1]]

    df["stressDelta"] = df["stressGroundTruth"] - df["stressSimulated"]
    sumOfDeltas = np.sum(np.abs(df["stressDelta"]))
    sumOfStressGroundTruth = np.sum(df["stressGroundTruth"])

    return float(sumOfDeltas / sumOfStressGroundTruth)
