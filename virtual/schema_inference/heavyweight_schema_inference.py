import duckdb
import pathlib
import pandas as pd
import virtual.utils
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
  def __init__(self, data: pd.DataFrame | pathlib.Path | virtual.utils.URLPath):
    assert isinstance(data, (pd.DataFrame | pathlib.Path, virtual.utils.URLPath))
    self.data = data

  def infer(self, nrows=None):
    # Be verbose.
    print(f'Running schema inference..')

    # Infer the schema via DuckDB.
    inferer = LWSchemaInferer(self.data)
    _, col_types = inferer.infer(nrows=nrows)

    # Is `data` a path?
    df = None
    if isinstance(self.data, pathlib.Path):
      if self.data.suffix == '.csv':
        # TODO: Do this better.
        # Infer the csv dialect.
        csv_dialect = virtual.utils.infer_dialect(self.data)

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
        pass
        # df = duckdb.read_parquet(str(self.data)).fetchdf()
      else:
        assert 0, 'File format not (yet) supported.'
    elif isinstance(self.data, virtual.utils.URLPath):
      assert self.data.suffix == '.parquet'
      pass
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
      assert df is not None
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

    # TODO: There is a problem with Redset here.
    # If we load directly into a dataframe, we get a different precision for `mbytes_spilled`.

    # The schema to be collected.
    schema = []
    if df is not None:
      for index in range(len(col_types)):
        col_dict = {
          'name': col_types[index]['name'],
          'type' : col_types[index]['type'],
          'null': check_for_null(col_types[index]['name']),
          'scale' : 0,
          'precision' : 0
        }

        # Double?
        if virtual.utils.is_fp(col_types[index]['type']):
          col_dict['precision'], col_dict['scale'] = calculate_precision_and_scale(col_types[index]['name'])

        # Add to the schema.
        schema.append(col_dict)
    else:
      assert isinstance(self.data, (pathlib.Path, virtual.utils.URLPath))
      assert self.data.suffix == '.parquet'
      cns = list(map(lambda elem: elem['name'], col_types))
      double_cns = list(map(lambda elem: elem['name'], filter(lambda elem: virtual.utils.is_fp(elem['type']), col_types)))

      # Define the NULL counts.
      null_counts = list(map(lambda cn: f"sum(case when \"{cn}\" is null then 1 else 0 end) as \"{cn}_null_count\"", cns))

      # Define the left precisions.
      left_precisions = list(map(lambda cn:f"""max(
          case
            when \"{cn}\" is null then 0
            else length(split_part(cast("{cn}" as varchar), '.', 1)) - (case when \"{cn}\" < 0 then 1 else 0 end)
          end
        ) as \"{cn}_left_precision\"""",
        double_cns
      ))

      # Define the right precisions.
      right_precisions = list(map(lambda cn:f"""max(
          case
            when \"{cn}\" is null then 0
            else length(nullif(split_part(cast(\"{cn}\" as varchar), '.', 2), ''))
          end
        ) as \"{cn}_right_precision\"""",
        double_cns
      ))

      # Be careful here since we might have no double-typed columns.
      limit_clause = ''
      if nrows is not None:
        limit_clause = f'limit {nrows}'
      sql_query = f"""
        select
          count(*),
          {', '.join(null_counts)}
          {(',' + ', '.join(left_precisions)) if double_cns else ''}
          {(',' + ', '.join(right_precisions)) if double_cns else ''}
        from read_parquet('{str(self.data)}')
        {limit_clause}
      """

      ret = duckdb.sql(sql_query).fetchone()

      # Complete the schema.
      table_size = ret[0]
      num_double_sofar = 0
      for index in range(len(col_types)):
        null_count = ret[index + 1]
        scale, precision = 0, 0
        if virtual.utils.is_fp(col_types[index]['type']):
          # Take the precisions.
          left_precision = ret[1 + len(col_types) + num_double_sofar]
          right_precision = ret[1 + len(col_types) + len(double_cns) + num_double_sofar]

          # Define the scale and the precision.
          precision = left_precision + right_precision
          scale = right_precision

          # Increase the counter.
          num_double_sofar += 1

        # And set.
        col_dict = {
          'name': col_types[index]['name'],
          'type' : col_types[index]['type'],
          'null': {
            'any' : null_count > 0,
            'all' : null_count == table_size
          },
          'scale' : scale,
          'precision' : precision
        }
        schema.append(col_dict)

    # And return it.
    return schema