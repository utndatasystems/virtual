from schema_inference.lightweight_schema_inference import LWSchemaInferer
from schema_inference.heavyweight_schema_inference import HWSchemaInferer
import virtual.utils
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
  cat2cns = {
    'num' : [],
    'date' : [],
    'timestamp' : [],
    'time' : [],
    'other' : [],
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

    if virtual.utils.is_num_virtualizable(sql_type):
      cat2cns['num'].append(cn)
    elif sql_type.lower() in ['date']:
      cat2cns['date'].append(cn)
    elif sql_type.lower() in ['timestamp']:
      cat2cns['timestamp'].append(cn)
    elif sql_type.lower() in ['time']:
      cat2cns['time'].append(cn)
    elif sql_type.lower() in ['datetime']:
      cat2cns['other'].append(cn)
    elif sql_type.lower() in ['varchar', 'char']:
      cat2cns['string'].append(cn)
    elif sql_type.lower() in ['boolean']:
      cat2cns['boolean'].append(cn)
    else:
      assert 0

    # Add to the column names.
    # NOTE: This is the original order in the dataframe / file.
    column_names.append(cn)

  # Infer the csv dialect (if any).
  csv_dialect = None
  if isinstance(data, pathlib.Path):
    if data.suffix == '.csv':
      csv_dialect = virtual.utils.infer_dialect(data)
  elif isinstance(data, virtual.utils.URLPath):
    # TODO: Support csv.
    pass
  elif isinstance(data, pd.DataFrame):
    # Default.
    csv_dialect = virtual.utils.get_default_csv_dialect()
  else:
    assert 0

  # And return.
  return column_names, csv_dialect, has_header, cat2cns