import duckdb
import pathlib
import pandas as pd
from virtual.utils import infer_dialect

def format_col_name(col_name):
  if col_name.startswith('"') and col_name.endswith('"'):
    return col_name
  return f'"{col_name}"'

class LWSchemaInferer:
  def __init__(self, data: pd.DataFrame | pathlib.Path):
    assert isinstance(data, (pd.DataFrame, pathlib.Path))
    self.data = data
    
  def infer(self, nrows=None):
    # Start by defining the query
    query = None
    if isinstance(self.data, pathlib.Path):
      # Infer the dialect.
      dialect = infer_dialect(self.data)

      # Specify the query.
      query = f"select * from read_csv('{self.data}', header=true, delim='{dialect['delimiter']}', quote='{dialect['quotechar']}'" \
        + (f", sample_size={nrows}" if nrows is not None else ", sample_size=-1") \
        + ", nullstr=['NULL', 'null', '', 'None'])"
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

    type_mapping = {columns[i]: str(column_types[i]) for i in range(len(columns))}  
    return type_mapping, has_header
