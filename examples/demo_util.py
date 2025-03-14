import requests
import os

def get_file_size(parquet_filepath):
  if parquet_filepath.startswith('s3://'):
    url = parquet_filepath.replace('s3://', 'https://s3.amazonaws.com/')
    response = requests.head(url)
    if response.status_code == 200:
      return int(response.headers.get('Content-Length', 0))
    else:
      raise Exception(f'Error: {response.status_code} - {response.reason}')
  else:
    return os.path.getsize(parquet_filepath)