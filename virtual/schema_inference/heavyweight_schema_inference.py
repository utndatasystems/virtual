import duckdb
import pathlib
import pandas as pd
from virtual.utils import infer_dialect
from .lightweight_schema_inference import LWSchemaInferer

class TypeChecker:
  def __init__(self):
    self.reset()
    pass

  def reset(self):
    self.left_precision, self.right_precision = 0, 0
    pass

  def _is_bool(self, value):
    try:
      return (len(value) == 4 or len(value) == 5) and value.lower() in ['true', 'false', 't', 'f']
    except:
      return False

  def _is_integer(self, value):
    # Check if any `.` in the string representation.
    if '.' in str(value):
      return False
    
    # Otherwise, try a cast.
    try:
      result = int(value)
      return result < 2**31 - 1 and result > -2**31
    except:
      return False
      
  def _is_bigint(self, value):
    # Check if any `.` in the string representation.
    if '.' in str(value):
      return False
    
    # Otherwise, try a cast.
    try:
      int(value)
      return True
    except:
      return False
      
  def _is_float(self, value):
    # NOTE: We no longer use `DecimalTuple`.
    # This is because for some numbers it even increased the number of decimals.

    # Calculate the precision.
    # NOTE: We also need to account for the possible minus sign in front of `x`.
    def calculate_precision(x):
      if x.startswith('-'):
        return len(x) - 1
      return len(x)

    try:
      # Convert to float.
      float(value)

      # In case we have an integer.
      # NOTE: This also includes the case in which we have an integer.
      if len(value.split('.')) == 1:
        # Update only the left precision.
        self.left_precision = max(self.left_precision, calculate_precision(value))
        return True

      # Assert that we always have two parts. 
      if len(value.split('.')) > 2:
        return False
      
      # Count the left and right precision.
      # NOTE: The left part could start with a negative sign, so we have to take that into account.
      self.left_precision = max(self.left_precision, calculate_precision(value.split('.')[0]))
      self.right_precision = max(self.right_precision, calculate_precision(value.split('.')[1]))
      return True
    except:
      return False

  def _is_null(self, value, verbose=False):
    if verbose:
      print(value)
    return value is None or pd.isna(value)

class HWSchemaInferer:
  def __init__(self, data: pd.DataFrame | pathlib.Path):
    assert isinstance(data, (pd.DataFrame | pathlib.Path))
    self.data = data

  def infer(self, nrows=None):
    # Infer the schema via DuckDB.
    inferer = LWSchemaInferer(self.data)
    _, col_types = inferer.infer(nrows=nrows)

    print(self.data.suffix)

    # Is `data` a path?
    if isinstance(self.data, pathlib.Path):
      if self.data.suffix == '.csv':
        # TODO: Do this better.
        # Infer the csv dialect.
        csv_dialect = infer_dialect(self.data)

        # Read dataframe. NOTE: If `nrows` is `None`, then the entire file is read.
        # TODO: Replace by DuckDB?
        df = pd.read_csv(
          self.data,
          dtype=str,
          nrows=nrows,
          # If the file contains a header row, then you should explicitly pass header=0 to override the column names.
          header=0,
          names=[elem['name'] for elem in col_types],
          delimiter=csv_dialect['delimiter'],
          quotechar=csv_dialect['quotechar']
        )
      elif self.data.suffix == '.parquet':
        df = duckdb.read_parquet(str(self.data)).fetchdf()
      else:
        assert 0, 'File format not (yet) supported.'
    elif isinstance(self.data, pd.DataFrame):
      # Do we have a number of rows specified.
      if nrows is not None:
        df = self.data.iloc[:nrows]
      else:
        # If not, then take the entire data. That's also the default in `pd.read_csv`.
        df = self.data
    else:
      # Unreachable.
      assert 0

    # A type checker.
    type_checker = TypeChecker()

    # Check if `col_name` has NULLs.
    def check_for_null(col_name):
      tmp = df[col_name].isnull()
      return {
        'any' : bool(tmp.any()),
        'all' : bool(tmp.all())
      }

    # Calculate the precision and the scale of `col_name`.
    def calculate_precision_and_scale(col_name):
      # Reset.
      type_checker.reset()

      # Iterate the column.
      for _, value in enumerate(df[col_name]):
        # Skip NULL values.
        if type_checker._is_null(value):
          continue
        assert type_checker._is_float(str(value))

      return type_checker.left_precision + type_checker.right_precision, type_checker.right_precision

    schema = []
    for index in range(len(col_types)):
      col_dict = {
        # NOTE: We remove the trailing spaces so that the column name matching is more efficient in the actual code.
        'name': col_types[index]['name'],
        'type' : col_types[index]['type'],
        'null': check_for_null(col_types[index]['name']),
        'scale' : 0,
        'precision' : 0
      }
      if col_types[index]['type'] == 'DOUBLE':
        col_dict['precision'], col_dict['scale'] = calculate_precision_and_scale(col_types[index]['name'])

      # Add to the schema.
      schema.append(col_dict)

    # And return it.
    return schema