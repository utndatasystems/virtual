import requests

def get_public_s3_file_size(s3_filepath):
  assert s3_filepath.startswith('s3://')
  url = s3_filepath.replace('s3://', 'https://s3.amazonaws.com/')
  response = requests.head(url)
  if response.status_code == 200:
    return int(response.headers.get('Content-Length', 0))
  else:
    raise Exception(f'Error: {response.status_code} - {response.reason}')