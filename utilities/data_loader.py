"""read the data from csv and return the dataframe"""
import pandas as pd
from numpy.typing import NDArray

def load_data(file_path: str) -> tuple[NDArray, NDArray, NDArray, NDArray]:
  """ Load the data into numpy arrays.

  Load the data from csv, then flip the order before splitting it into numpy array.

  Args:
    file_path (str): file path of the csv file

  Returns:
    tuple(numpyArray, numpyArray, numpyArray, numpyArray): tuple of numpy arrays arranged in
      (prices pre-2020, dates pre-2020, prices from 2020+, dates from 2020-01-01+)
  Returns
  """
  df = pd.read_csv(file_path)
  # The order of data is descending (newest first, so flip with -1 indexing)
  df = df[::-1].reset_index(drop=True)
  df['date'] = pd.to_datetime(df['date'])
  split_idx = df[df['date'] >= '2020-01-01'].index[0]
  prices = df['close'].to_numpy()
  dates = df['date'].to_numpy()
  return prices[:split_idx], dates[:split_idx], prices[split_idx:], dates[split_idx:]