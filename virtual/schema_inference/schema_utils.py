from schema_inference.lightweight_schema_inference import LWSchemaInferer
from schema_inference.heavyweight_schema_inference import HWSchemaInferer
from virtual.utils import infer_dialect, get_default_csv_dialect
import pandas as pd
import pathlib
import re

# Generate the schema by the heavyweight variant (it takes a while if there are many columns).
def generate_schema(data: pd.DataFrame | pathlib.Path, nrows=None):
  # Convert to an actual path.
  if isinstance(data, str):
    data = pathlib.Path(data)
  
  # Build the inferer.
  inferer = HWSchemaInferer(data)

  # And return.
  return inferer.infer(nrows)

def map_sql_to_pandas(sql_type):
  type_mapping = {
    'timestamp': 'datetime64[ns]',
    'datetime' : 'datetime64[ns]',
    'date' : 'datetime64[ns]',
    'time' : 'datetime64[ns]',
    'decimal': 'float64',
    'double': 'float64',
    # TODO: Fix this later. The issue is that DuckDB interprets as `BOOLEAN` any binary column.
    'boolean': 'str',
    'varchar': 'str',
    'char' : 'str',
    # TODO: is this a problem?
    'smallint': pd.Int16Dtype(),
    'int': pd.Int32Dtype(),
    'bigint': pd.Int64Dtype()
    # Add other types as needed
  }
    
  # Extract the base SQL type (ignore size specifiers in parentheses)
  base_type = re.match(r'^\w+', sql_type.lower()).group(0)

  # If the base type is decimal, handle precision and scale
  if base_type == 'decimal':
    # Note: `pandas`` will use float64 for decimals
    return 'float64'

  return type_mapping.get(base_type, 'object')

def handle_schema(data: pd.DataFrame | pathlib.Path, nrows=None):
  dtype_mapping = {}
  column_names, date_cns, string_cns, boolean_cns = [], [], [], []

  # Read the colum types.
  inferer = LWSchemaInferer(data)

  # And infer
  schema, has_header = inferer.infer(nrows=nrows)

  # Build the type mapping.
  # TODO: I don't get why we still need this one.
  # TODO: I mean, if we'll switch to DuckDB, then there is no reason, why we should still do this.
  for col in schema:
    column_names.append(col)
    typ = schema[col]

    # TODO: There are still some hacks we did. 
    pandas_type = map_sql_to_pandas(typ)

    if pandas_type == 'datetime64[ns]':
      date_cns.append(col)
    else:
      dtype_mapping[col] = pandas_type
      if pandas_type == 'str':
        string_cns.append(col)
      elif pandas_type == 'boolean':
        boolean_cns.append(col)

  # Infer the delimiter.
  if isinstance(data, pathlib.Path):
    csv_dialect = infer_dialect(data)
  elif isinstance(data, pd.DataFrame):
    # Default.
    csv_dialect = get_default_csv_dialect()
  else:
    assert 0

  # And return.
  return column_names, dtype_mapping, csv_dialect, has_header, {
    'date' : date_cns,
    'string' : string_cns,
    'boolean' : boolean_cns
  }