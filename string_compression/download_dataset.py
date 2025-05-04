import json
import os
from huggingface_hub import snapshot_download
from datetime import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def write_parquet(df, path, PARQUET_COMPRESSION_TYPE="snappy"):
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path, compression=PARQUET_COMPRESSION_TYPE)

def get_df(path, max_rows=1_000_000):
    parts = []
    total_rows = 0

    for fname in sorted(os.listdir(path)):
        df_part = pd.read_parquet(os.path.join(path, fname))
        n = len(df_part)

        if total_rows + n >= max_rows:
            # Only take what we still need to reach the cap
            df_part = df_part.iloc[: max_rows - total_rows]
            parts.append(df_part)
            break

        parts.append(df_part)
        total_rows += n

        if total_rows >= max_rows:
            break

    return pd.concat(parts, ignore_index=True)

def download_dataset(dataset_id, config):
    if dataset_id == "HuggingFaceFW/fineweb":
        pattern = f"{config.split('-')[0]}/{config.split('-')[1]}/*"
    elif dataset_id == "wikimedia/wikipedia":
        pattern = f"{config}/*"
    else:
        print(f"Unknown dataset_id: {dataset_id}")
        return
    snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        local_dir=f"./datasets/{dataset_id}",
        cache_dir="./datasets/.cache",
        allow_patterns=[pattern]
    )

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

compression_methods = ["snappy", "gzip", "brotli", "lz4", "zstd"]
tmp_parquet = "tmp.parquet"
configs_file = "config.json"
config_dict = json.load(open(configs_file, "r"))
for info in config_dict:
    print(f"Name: {info}")
    for config in config_dict[info]["configs"]:
        print(f"  Config: {config['file']}")
        print(f"    status: {config['status']}")

        if config["status"] == "not yet started":
            download_dataset(info, config["file"])
            config["status"] = "downloaded"
            config["download_datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            config["path"] = f"./datasets/{info}/{config['file']}"
        elif config["status"] == "downloaded":
            if "size_original" not in config:
                config["size_original"] = get_size(config["path"])
                df = get_df(config["path"])
                for compression_method in compression_methods:
                    write_parquet(df, tmp_parquet, PARQUET_COMPRESSION_TYPE=compression_method)
                    config["size_" + compression_method] = get_size(tmp_parquet)
                config["rows"] = len(df)
            pass
    os.system(f"rm -rf ./datasets/{info}/.cache")

# remove the cache
os.system("rm -rf ./datasets/.cache")
# remove the tmp parquet file
os.system(f"rm -rf {tmp_parquet}")

with open(configs_file, "w") as f:
    json.dump(config_dict, f, indent=2)