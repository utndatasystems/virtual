import os
import json
import yaml
import utils

def schema_to_yaml(schema):
  dmap = {
    "VARCHAR": "string",
    "BIGINT": "integer",
    "SMALLINT": "integer",
    "DOUBLE": "double",
    "TIMESTAMP": "string",
    "BOOLEAN": "integer",
    "DATE": "string",
    "TIME": "string"
  }
  # schema = utils._read_json(schema)
  yaml_data = {"columns": None}
  columns_tmp = []
  for col in schema:
    columns_tmp.append({"name": col["name"], "type": dmap[col["type"]]})

  yaml_data["columns"] = columns_tmp
  return yaml_data

def generate_yaml_doc(yaml_file, schema):
  yaml_data = schema_to_yaml(schema)
  file = open(yaml_file, 'w', encoding='utf-8')
  yaml.dump(yaml_data, file)
  file.close()

def convert_csv_to_btrblocks(csv_path: str, format_path: str, schema: str | list, convertor_path = '../../btrblocks/build/'):
  if isinstance(schema, str):
    schema = utils._read_json(schema)
  yaml_name = "tmp.yaml"
  extra_path = os.path.join(format_path, 'stats')
  btr_path = os.path.join(format_path, 'btr/')
  binary_path = os.path.join(format_path, 'binary/')
  yaml_path = os.path.join(extra_path, yaml_name)
  os.makedirs(format_path, exist_ok=True)
  os.makedirs(extra_path, exist_ok=True)
  # print(f"format_path {format_path}")
  # print(f"yaml_path {yaml_path}")

  generate_yaml_doc(yaml_path, schema)
  os.system(f"rm -rf {btr_path}")
  os.system(f"rm -rf {binary_path}")
  try:
    os.system(f'{os.path.join(convertor_path, 'csvtobtr')} -create_btr -csv {csv_path} -yaml {yaml_path} -binary {binary_path} -create_binary -btr {btr_path} -stats {os.path.join(extra_path, 'stats.txt')} -compressionout {os.path.join(extra_path, 'compression.txt')}')
  except:
    print(f"Error processing {csv_path}.")

  os.system(f"rm -rf {yaml_path}")
  stas = get_status(extra_path)
  os.system(f"rm -rf {binary_path}")
  return stas

def get_status(extra_path, csv_name = None):
  if csv_name is None:
    stas = {
      'total': {},
      'columns': {}
    }
  else:
    stas = {
      'filename': csv_name+'.csv',
      'total': {},
      'columns': {}
    }
  with open(f"{os.path.join(extra_path, 'stats.txt')}", 'r') as f:
    rows = f.read()
  elements = rows.split('\n')[1:-1]
  for i, row in enumerate(elements):
    if i < len(elements) - 1:
      row_e = row.rsplit(',', 2)
      stas['columns'][row_e[0].split('_', 1)[1]] = {'btrblock-uncompressed': int(row_e[1]), 'btrblock-compressed': int(row_e[2])}
    else:
      row_e = row.split(',')
      stas['total'] = {'btrblock-uncompressed': int(row_e[1]), 'btrblock-compressed': int(row_e[2])}
          
  return stas