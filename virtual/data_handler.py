import pandas as pd
import pathlib
import duckdb

# Custom helper scripts.
from virtual.schema_inference.schema_utils import handle_schema
import utils

USE_DUCKDB_PARSER = False

class DataWrapper:
  def __init__(self, data: pd.DataFrame | pathlib.Path | utils.URLPath, nrows=None):
    self.data = data
    self.nrows = nrows

  def inspect_columns(self):
    # Inspect the schema.
    self.column_names, self.csv_dialect, self.has_header, self.type2cns = handle_schema(self.data)

    # Select the virtualizable column indices.
    self.v_cols = self.get_v_columns()

  def get_rank(self, category, idxs):
    if isinstance(idxs, list):
      return [self.v_cols[category]['indices'].index(idx) for idx in idxs]
    return self.v_cols[category]['indices'].index(idxs)

  def sample(self, category, sample_size=None):
    df = None
    if isinstance(self.data, pathlib.Path):
      # TODO: Move this support later, since we can directly sample from the CSV file.
      if self.data.suffix == '.csv':
        df = self.parse(category)
    elif isinstance(self.data, pd.DataFrame):
      df = self.parse(category)

    # Have at least a sample size.
    if sample_size is None:
      sample_size = 1_000

    # Sample now.
    if df is not None:
      # Take care of NULLs.
      df_without_null = df.dropna(subset=self.v_cols[category]['names'])

      # Note: If this doesn't hold, then it's a bit problematic for linear regression.
      # This is because this is an indeterminate system.
      if len(df_without_null) < len(self.v_cols[category]['names']):
        df_without_null = df.fillna(0)

      # Handle special cases.
      if len(df) == 0:
        sample = df_without_null
        ratio = 1.0
      else:
        ratio = sample_size / len(df_without_null)

      # (Trivially) upper-bound the ratio. This happens if the sample size is larger than the data itself.
      ratio = min(ratio, 1.0)

      # And sample.
      sample = df_without_null.sample(frac=ratio, replace=False, random_state=0)
    else:
      # We can extract the sample from the data itself.
      if isinstance(self.data, pathlib.Path):
        if self.data.suffix == '.parquet':
          sample = utils.sample_parquet_file(self.data, self.nrows, sample_size, category, self.v_cols[category]['names'])
      elif isinstance(self.data, utils.URLPath):
        if self.data.suffix == '.parquet':
          sample = utils.sample_parquet_file(self.data, self.nrows, sample_size, category, self.v_cols[category]['names'])

    # Return the sample.
    assert sample is not None
    return sample

  def parse(self, category):    
    # Read the CSV with the dtype mapping and date parsing, and without headers
    # TODO: Why does the other need a `dtype_mapping`?
    # TODO: I think this was because we always used SQL to infer the types.
    # TODO: So we can remove that part, also in the schema inference.

    # Specify the column indices we want to parse.
    if isinstance(self.data, pathlib.Path):
      if USE_DUCKDB_PARSER:
        # Construct the DuckDB query
        query = utils.build_query(
          select_stmt=', '.join(self.v_cols[category]['indices']),
          input=self.data,
          header=self.has_header,
          delim=self.csv_dialect['delimiter'],
          quotechar=self.csv_dialect['quotechar'],
          nullstr=utils.get_csv_null_options(),
          sample_size=self.nrows
        )
        
        # Add row limit if specified.
        # NOTE: This time we _really_ have to read the data.
        if self.nrows:
          query += f" LIMIT {self.nrows}"

        # Fetch the `df`.
        df = duckdb.query(query).to_df()
      else:
        df = pd.read_csv(
          self.data,
          names=self.column_names,
          # TODO: There was indeed a problem here.
          # dtype=dtype_mapping,
          # TODO: Removed this, since it's really slow and we don't need date columns anyway.
          # parse_dates=date_columns,
          usecols=self.v_cols[category]['indices'],
          header=self.has_header,
          delimiter=self.csv_dialect['delimiter'],
          quotechar=self.csv_dialect['quotechar'],
          nrows=self.nrows
        )
    elif isinstance(self.data, pd.DataFrame):
      df = self.data
      if self.v_cols[category]['indices'] is not None:
        df = df.iloc[:, self.v_cols[category]['indices']]
      
      # Limit the number of rows.
      if self.nrows is not None:
        df = df.head(self.nrows)
    else:
      # Unsupported data source. Unreachable for now.
      assert 0
    
    # NOTE: Very important. Update the column names.
    # NOTE: Not needed anymore since we always set the valid columns.
    # self.column_names = df.columns

    # And return the dataframe.
    return df
    
  def get_v_columns(self):
    v_cols = dict()
    for key in self.type2cns:
      v_cols[key] = {
        'indices' : [self.column_names.index(cn) for cn in self.type2cns[key]],
        'names' : self.type2cns[key]
      }
    return v_cols