import pandas as pd
import os
import json
from urllib.parse import quote
import pyarrow as pa
import pyarrow.parquet as pq

def write_parquet(df, path, PARQUET_COMPRESSION_TYPE="zstd"):
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path, compression=PARQUET_COMPRESSION_TYPE)


def compare_dataframes(df1, df2):
    # Align columns by name
    df1_sorted = df1.sort_index(axis=1)
    df2_sorted = df2[df1_sorted.columns]  # reorder df2 to match df1

    # Check shape first
    if df1_sorted.shape != df2_sorted.shape:
        print("DataFrames have different shapes:", df1_sorted.shape, df2_sorted.shape)
        return

    # Create a boolean DataFrame of element-wise comparisons
    diff = df1_sorted != df2_sorted

    if not diff.any().any():
        print("DataFrames are equal")
    else:
        print("Differences found at these locations:")
        differing_cells = diff.stack()[diff.stack()]  # True where differences exist
        for (idx, col) in differing_cells.index:
            print(f"- Row {idx}, Column '{col}': df1 = {df1_sorted.at[idx, col]!r}, df2 = {df2_sorted.at[idx, col]!r}")

def get_size(start_path = '.'):
    # check is it a file or directory
    if os.path.isfile(start_path):
        return os.path.getsize(start_path)
    total = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total

def get_df(path):
    file_list = os.listdir(path)
    print(file_list)
    dfs = [pd.read_parquet(os.path.join(path, file)) for file in file_list]
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df
        

os.makedirs("datasets_compress/wikimedia/wikipedia", exist_ok=True)

dataset_id = "wikimedia/wikipedia"
configs_file = "config.json"
compression_method = "zstd" # "snappy", "gzip", "brotli", "lz4", "zstd"
config_dict = json.load(open(configs_file, "r"))
prefixs = {
    '20231101.zh-classical': "https://zh-classical.wikipedia.org/wiki/", 
    '20231101.ab': "https://ab.wikipedia.org/wiki/", 
    '20231101.af': "https://af.wikipedia.org/wiki/", 
    '20231101.azb': "https://azb.wikipedia.org/wiki/", 
    '20231101.bg': "https://bg.wikipedia.org/wiki/",
    '20231101.ady': "https://ady.wikipedia.org/wiki/",
    '20231101.bg': "https://bg.wikipedia.org/wiki/",
    '20231101.ar': "https://ar.wikipedia.org/wiki/",
    '20231101.de': "https://de.wikipedia.org/wiki/",
    '20231101.ceb': "https://ceb.wikipedia.org/wiki/",
    '20231101.nl': "https://nl.wikipedia.org/wiki/"
}

total_size = 0
total_compress_size = 0
for config in config_dict[dataset_id]["configs"]:
    if "compress" not in config:
        config["compress"] = []

    if len(config["compress"]) == 0:
        config["compress"].append({"method": "url=offset+transfer(title)"})
    compress_config = config["compress"][0]
    print(f"  Config: {config['file']}")
    config_name = config["file"]
    df = get_df(f"datasets/{dataset_id}/{config_name}/")
    compress_df = df.drop(columns=["url"])
    write_parquet(compress_df, f"datasets_compress/{dataset_id}/{config_name}.parquet", PARQUET_COMPRESSION_TYPE=compression_method)
    compress_df = pd.read_parquet(f"datasets_compress/{dataset_id}/{config_name}.parquet")
    size = config[f"size_{compression_method}"]
    compression_size = get_size(f"datasets_compress/{dataset_id}/{config_name}.parquet")
    print(f"Size of the dataset: {size / (1024):.2f} kB")
    print(f"Size of the compression: {compression_size / (1024):.2f} kB")
    print(f"Compression ratio: {((size-compression_size) / size) * 100:.2f} %")
    compress_config[f"size_compression_{compression_method}"] = compression_size
    total_size += size
    total_compress_size += compression_size
    compress_df["url"] = compress_df["title"].apply(lambda x: prefixs[config_name] + quote(x))
    compare_dataframes(df, compress_df)
    config["compress"][0] = compress_config
    

print(f"Total size of the dataset ({compression_method}): {total_size / (1024):.2f} kB")
print(f"Total size of the compression ({compression_method}): {total_compress_size / (1024):.2f} kB")
print(f"Total compression ratio ({compression_method}): {((total_size-total_compress_size) / total_size) * 100:.2f} %")

with open(configs_file, "w") as f:
    json.dump(config_dict, f, indent=2)




