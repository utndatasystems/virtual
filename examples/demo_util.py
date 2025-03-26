import requests
import os

def get_file_size(parquet_filepath):
  def size_impl(url):
    response = requests.head(url)
    if response.status_code == 200:
      return int(response.headers.get('Content-Length', 0))
    else:
      raise Exception(f'Error: {response.status_code} - {response.reason}')

  if parquet_filepath.startswith('s3://'):
    url = parquet_filepath.replace('s3://', 'https://s3.amazonaws.com/')
    return size_impl(url)
  elif parquet_filepath.startswith('https://'):
    return size_impl(parquet_filepath)
  else:
    return os.path.getsize(parquet_filepath)