import pandas as pd
import pathlib
import duckdb

# Custom helper scripts.
from virtual.schema_inference.schema_utils import handle_schema
from virtual.utils import build_query, get_csv_null_options

USE_DUCKDB_PARSER = False

class DataParser:
  def __init__(self, data: pd.DataFrame | pathlib.Path, nrows=None):
    self.data = data
    self.nrows = nrows

  def parse(self):    
    # Parse the schema.

    # TODO: Indeed, do we still use `dtype_mapping`?
    self.column_names, dtype_mapping, csv_dialect, has_header, self.type_cns = handle_schema(self.data)

    # Read the CSV with the dtype mapping and date parsing, and without headers
    # TODO: Why does the other need a `dtype_mapping`?
    # TODO: I think this was because we always used SQL to infer the types.
    # TODO: So we can remove that part, also in the schema inference.

    # Specify the column indices we want to parse.
    # TODO: Also remove strings.
    use_col_idxs = [index for index, cn in enumerate(self.column_names) if cn not in self.type_cns['date']]
    if isinstance(self.data, pathlib.Path):
      if USE_DUCKDB_PARSER:
        # Take the selected columns.
        selected_columns = (
          [self.column_names[idx] for idx in use_col_idxs]
          if use_col_idxs else "*"
        )

        # Construct the DuckDB query
        query = build_query(
          select_stmt=', '.join(selected_columns),
          input=self.data,
          header=has_header,
          delim=csv_dialect['delimiter'],
          quotechar=csv_dialect['quotechar'],
          nullstr=get_csv_null_options(),
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
          usecols=use_col_idxs,
          header=has_header,
          delimiter=csv_dialect['delimiter'],
          quotechar=csv_dialect['quotechar'],
          nrows=self.nrows
        )
    elif isinstance(self.data, pd.DataFrame):
      df = self.data
      if use_col_idxs is not None:
        df = df.iloc[:, use_col_idxs]
      
      # Limit the number of rows.
      if self.nrows is not None:
        df = df.head(self.nrows)
    else:
      # Unsupport data source. Unreachable for now.
      assert 0
    
    # Note: Very important. Update the column names.
    self.column_names = df.columns

    # And return the dataframe.
    return df
    
  def compute_valid_column_indices(self, df):
    valid_column_indices = []
    for i, cn in enumerate(df.columns):
      if cn in self.type_cns['date'] or cn in self.type_cns['string'] or cn in self.type_cns['boolean']:
        continue
      valid_column_indices.append(i)
    return valid_column_indices

  def extract_sample(self, df, sample_size=None):
    valid_column_indices = self.compute_valid_column_indices(df)
    valid_column_names = [self.column_names[i] for i in valid_column_indices]

    df_without_null = df.dropna(subset=valid_column_names)

    # Note: If this doesn't hold, then it's a bit problematic for linear regression.
    # This is because this is an indeterminate system.
    if len(df_without_null) < len(valid_column_indices):
      df_without_null = df.fillna(0)

    if sample_size is None:
      sample_size = 1_000

    # Handle special cases.
    if len(df) == 0:
      sample = df_without_null
      ratio = 1
    else:
      ratio = sample_size / len(df_without_null)

    if ratio > 1:
      ratio = 1

    # And sample.
    sample = df_without_null.sample(frac=ratio, replace=False, random_state=0)

    # Return the sample.
    return sample
