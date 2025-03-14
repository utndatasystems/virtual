from schema_inference.lightweight_schema_inference import LWSchemaInferer
from schema_inference.heavyweight_schema_inference import HWSchemaInferer
from virtual.utils import infer_dialect, get_default_csv_dialect
import pandas as pd
import pathlib

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
  # The column categories.
  type_categories = {
    'date' : [],
    'string' : [],
    # TODO: Fix this later. The issue is that DuckDB interprets as `BOOLEAN` any binary column.
    'boolean' : []
  }

  # Read the colum types.
  # TODO: Are we calling this twice?
  inferer = LWSchemaInferer(data)

  # And infer.
  has_header, col_types = inferer.infer(nrows=nrows)

  # Categorize the column types.
  column_names = []
  for index in range(len(col_types)):
    cn = col_types[index]['name']
    sql_type = col_types[index]['type']

    if sql_type.lower() in ['datetime', 'timestamp', 'time', 'date']:
      type_categories['date'].append(cn)
    elif sql_type.lower() in ['varchar', 'char']:
      type_categories['string'].append(cn)
    elif sql_type.lower() in ['boolean']:
      type_categories['boolean'].append(cn)

    # Add to the column names.
    # NOTE: This is the original order in the dataframe / file.
    column_names.append(cn)

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
  return column_names, csv_dialect, has_header, type_categories