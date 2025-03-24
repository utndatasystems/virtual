import duckdb
import pathlib
import pandas as pd
import virtual.utils

def format_col_name(col_name):
  if col_name.startswith('"') and col_name.endswith('"'):
    return col_name
  return f'"{col_name}"'

class LWSchemaInferer:
  def __init__(self, data: pd.DataFrame | pathlib.Path | virtual.utils.URLPath):
    assert isinstance(data, (pd.DataFrame, pathlib.Path, virtual.utils.URLPath))
    self.data = data
    
  def infer(self, nrows=None):
    # Start by defining the query
    query = None
    if isinstance(self.data, pathlib.Path):
      if self.data.suffix == '.csv':
        # TODO: We should use the schema if possible. Since this is really slow for large CSV files.
        # Infer the dialect.
        dialect = virtual.utils.infer_dialect(self.data)

        # Specify the query.
        query = f"select * from read_csv('{self.data}', header=true, delim='{dialect['delimiter']}', quote='{dialect['quotechar']}'" \
          + (f", sample_size={nrows}" if nrows is not None else ", sample_size=-1") \
          + ", nullstr=['NULL', 'null', '', 'None'])"
      elif self.data.suffix == '.parquet':
        query = f"""
          select *
          from read_parquet('{self.data}')
          limit 0;
        """
      else:
        assert 0, 'File format not (yet) supported.'
    elif isinstance(self.data, virtual.utils.URLPath):
      if self.data.suffix == '.parquet':
        query = f"""
          select *
          from read_parquet('{str(self.data)}')
          limit 0;
        """
      else:
        assert 0, 'We don\'t support yet reading this format from an URL.'
    elif isinstance(self.data, pd.DataFrame):
      duckdb.register("mydf", self.data)
      query = f'SELECT * FROM mydf' + (f'LIMIT {nrows}' if nrows is not None else '')
    else:
      assert 0
      
    # Execute the query.
    result = duckdb.sql(query)
    column_types = result.types
    columns = result.columns

    # Determine if the data has a header
    # TODO: Fix this since might not be reliable.
    if columns[0] != "column0":
      has_header = 0
    else:
      has_header = None

    col_types = []
    for index in range(len(columns)):
      column_name = columns[index]

      # Strip the name of double-quotes, since pd.DataFrame puts them by default.
      if isinstance(self.data, pd.DataFrame):
        if column_name.startswith('"') and column_name.endswith('"'):
          column_name = column_name[1:-1]

      # And add.
      col_types.append({
        'name' : column_name,
        'type' : str(column_types[index])
      })
    return has_header, col_types