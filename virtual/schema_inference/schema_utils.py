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

def handle_schema(data: pd.DataFrame | pathlib.Path, nrows=None):
  column_names = []

  # The column categories.
  column_categories = {
    'date' : [],
    'string' : [],
    # TODO: Fix this later. The issue is that DuckDB interprets as `BOOLEAN` any binary column.
    'boolean' : []
  }

  # Read the colum types.
  # TODO: Are we calling this twice?
  inferer = LWSchemaInferer(data)

  # And infer
  schema, has_header = inferer.infer(nrows=nrows)

  # Categorize the column names.
  for col in schema:
    column_names.append(col)
    sql_type = schema[col]

    if sql_type in ['datetime', 'timestamp', 'time', 'date']:
      column_categories['date'].append(col)
    elif sql_type in ['varchar', 'char']:
      column_categories['string'].append(col)
    elif sql_type in ['boolean']:
      column_categories['boolean'].append(col)

  # Infer the csv dialect (if any).
  csv_dialect = None
  if isinstance(data, pathlib.Path):
    if data.suffix == '.csv':
      csv_dialect = infer_dialect(data)
  elif isinstance(data, pd.DataFrame):
    # Default.
    csv_dialect = get_default_csv_dialect()
  else:
    assert 0

  # And return.
  return column_names, csv_dialect, has_header, column_categories