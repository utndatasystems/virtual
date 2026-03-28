
import pandas as pd
import os
import json
import pyarrow as pa
import argparse
from sklearn.linear_model import LinearRegression
from utils import write_parquet, get_df, compare_dataframes, get_size

def main():
    """
    This script compresses the Fineweb dataset using different methods.
    The compression method is selected using a command-line argument.
    The script reads a configuration file, processes the dataset,
    applies the selected compression, and saves the result.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Compress Fineweb dataset.')
    parser.add_argument('compression_method', type=int, choices=range(1, 9), help='Compression method to use (1-8)')
    args = parser.parse_args()

    # --- Configuration and Setup ---
    dataset_id = "HuggingFaceFW/fineweb"
    os.makedirs(f"datasets_compress/{dataset_id}", exist_ok=True)
    configs_file = "config.json"
    compression_methods = ["snappy", "gzip", "brotli", "lz4", "zstd"]
    config_dict = json.load(open(configs_file, "r"))

    # Initialize dictionaries to store size information
    total_size = {
        "snappy": 0,
        "gzip": 0,
        "brotli": 0,
        "lz4": 0,
        "zstd": 0
    }
    total_compress_size = {
        "snappy": 0,
        "gzip": 0,
        "brotli": 0,
        "lz4": 0,
        "zstd": 0
    }

    # --- Prefix Dictionary ---
    prefixs = {
        "id": "<urn:uuid:",
        "dump": "CC-MAIN-",
        "url": "http",
        "file_path": "s3://commoncrawl/crawl-data/CC-MAIN-"
    }

    # --- Data Processing Loop ---
    for config in config_dict[dataset_id]["configs"]:
        # Ensure the 'compress' key exists
        if "compress" not in config:
            config["compress"] = []

        # Add a new dictionary if the compression method has not been run before
        if len(config["compress"]) < args.compression_method:
            config["compress"].append({})
        compress_config = config["compress"][args.compression_method - 1]
        
        print(f"  Config: {config['file']}")
        config_name = config["file"]
        
        # Load the dataset
        df = get_df(config["path"])

        # --- Compression and Evaluation Loop ---
        for compression_method in compression_methods:
            compress_df = df.copy()

            # --- Compression Logic ---
            if args.compression_method == 1:
                # Method 1: Drop the 'dump' column
                compress_config["function"] = "dump=file_path.split('/')[4].split('/')[0]"
                compress_df = compress_df.drop(columns=["dump"])
            elif args.compression_method == 2:
                # Method 2: Remove the dump from the file path
                compress_config["function"] = "file_path=file_path+dump+file_path"
                compress_df["file_path"] = compress_df.apply(lambda r: r.file_path.replace(r.dump, ""), axis=1)
            elif args.compression_method == 3:
                # Method 3: Combine all string columns into a single column
                compress_config["function"] = "combine all string columns"
                columns_names = []
                for col in compress_df.select_dtypes(include="object"):   # only string‑type columns
                    if "combine" not in compress_df.columns:
                        compress_df["combine"] = compress_df[col]
                    else:
                        compress_df["combine"] = compress_df[col] + compress_df["combine"]
                        compress_df[f"{col}_len"] = compress_df[col].str.len()
                    columns_names.append(col)
                compress_df = compress_df.drop(columns=columns_names)
            elif args.compression_method == 4:
                # Method 4: Use linear regression to predict token_count from text_len
                compress_config["function"] = "linear regression: text_len to token_count"
                compress_df["text_len"] = compress_df["text"].str.len()
                X = compress_df[["text_len"]]
                y = compress_df["token_count"]
                model = LinearRegression()
                model.fit(X, y)
                compress_df["predicted_token_count"] = model.predict(X)
                compress_df["predicted_token_count"] = compress_df["predicted_token_count"].round(0).astype(int)
                compress_df["token_count_offset"] = compress_df["token_count"] - compress_df["predicted_token_count"]
                compress_df = compress_df.drop(columns=["text_len", "predicted_token_count", "token_count"])
            elif args.compression_method == 5:
                # Method 5: Remove prefixes from specified columns
                compress_config["function"] = "remove prefixs"
                for key in prefixs.keys():
                    compress_df[key] = compress_df[key].apply(lambda x: x.replace(prefixs[key], "", 1))
            elif args.compression_method == 6:
                # Method 6: Split the 'id' column into five parts
                compress_config["function"] = "split the id"
                compress_df["id"] = compress_df["id"].apply(lambda x: str(x).replace("<urn:uuid:", "").replace(">", ""))
                compress_df["id_part1"] = compress_df["id"].apply(lambda x: x.split("-")[0])
                compress_df["id_part2"] = compress_df["id"].apply(lambda x: x.split("-")[1])
                compress_df["id_part3"] = compress_df["id"].apply(lambda x: x.split("-")[2])
                compress_df["id_part4"] = compress_df["id"].apply(lambda x: x.split("-")[3])
                compress_df["id_part5"] = compress_df["id"].apply(lambda x: x.split("-")[4])
                compress_df = compress_df.drop(columns=["id"])
            elif args.compression_method == 7:
                # Method 7: Concatenate 'text' and 'url' columns
                compress_config["function"] = "concat text and url"
                compress_df["combine"] = compress_df['url'] + compress_df['text']
                compress_df[f"url_len"] = compress_df['url'].str.len()
                compress_df = compress_df.drop(columns=["url", "text"])
            elif args.compression_method == 8:
                # Method 8: Concatenate text and the last part of the URL
                compress_config["function"] = "concat text and url_last_part, url = url_prefix + url_last_part"
                compress_df["url_split"] = compress_df["url"].str.split("/", n=3)
                compress_df["has_trailing_slash"] = compress_df["url"].str.endswith("/")
                compress_df["url_prefix"] = compress_df["url_split"].str[:3].str.join("/")
                compress_df["url_prefix"] = compress_df.apply(
                    lambda r: r["url_prefix"] + "/" if len(r["url_split"]) <= 3 and r["has_trailing_slash"] else r["url_prefix"],
                    axis=1
                )
                compress_df["url_path"] = compress_df["url_split"].str[3].fillna("")
                compress_df["combine"] = compress_df["url_path"] + compress_df["text"]
                compress_df["url_len"] = compress_df["url_path"].str.len()
                compress_df = compress_df.drop(columns=["url", "text", "url_split", "url_path", "has_trailing_slash"])

            # --- Save and Evaluate ---
            # Write the compressed dataframe to a parquet file
            write_parquet(compress_df, f"datasets_compress/{dataset_id}/{config_name}.parquet", PARQUET_COMPRESSION_TYPE=compression_method)
            
            # Get the original and compressed sizes
            size = config[f"size_{compression_method}"]
            compression_size = get_size(f"datasets_compress/{dataset_id}/{config_name}.parquet")
            
            # Print compression statistics
            print(f"Size of the dataset: {size / (1024):.2f} kB")
            print(f"Size of the compression: {compression_size / (1024):.2f} kB")
            print(f"Compression ratio: {((size-compression_size) / size) * 100:.2f} %")
            
            # Store the compressed size in the config
            compress_config[f"size_compression_{compression_method}"] = compression_size
            total_size[compression_method] += size
            total_compress_size[compression_method] += compression_size

            # --- Decompression and Verification (Commented Out) ---
            # if args.compression_method == 1:
            #     compress_df["dump"] = compress_df["file_path"].apply(lambda x: x.split("/")[4].split("/")[0])
            # elif args.compression_method == 2:
            #     compress_df["file_path"] = compress_df.apply(lambda r: r.file_path.replace("crawl-data/", f"crawl-data/{r.dump}"), axis=1)
            # elif args.compression_method == 3:
            #     reversed_columns_names = columns_names[::-1]
            #     for i, columns_name in enumerate(reversed_columns_names):
            #         if i == len(columns_names) - 1:
            #             compress_df[columns_name] = compress_df["combine"]
            #             compress_df = compress_df.drop(columns=["combine"])
            #         else:
            #             compress_df[columns_name] = compress_df.apply(lambda r: r["combine"][: r[f"{columns_name}_len"]], axis=1)
            #             compress_df["combine"] = compress_df.apply(lambda r: r["combine"][r[f"{columns_name}_len"]: ], axis=1)
            #             compress_df = compress_df.drop(columns=[columns_name+"_len"])
            # elif args.compression_method == 4:
            #     compress_df["text_len"] = compress_df["text"].str.len()
            #     X = compress_df[["text_len"]]
            #     compress_df["predicted_token_count"] = model.predict(X)
            #     compress_df["predicted_token_count"] = compress_df["predicted_token_count"].round(0).astype(int)
            #     compress_df["token_count"] = compress_df["token_count_offset"] + compress_df["predicted_token_count"]
            #     compress_df = compress_df.drop(columns=["text_len", "predicted_token_count", "token_count_offset"])
            # elif args.compression_method == 5:
            #     for key in prefixs.keys():
            #         compress_df[key] = compress_df[key].apply(lambda x: prefixs[key] + x)
            # elif args.compression_method == 6:
            #     compress_df["id"] = "<urn:uuid:" + compress_df["id_part1"] + "-" + compress_df["id_part2"] + "-" + compress_df["id_part3"] + "-" + compress_df["id_part4"] + "-" + compress_df["id_part5"] + ">"
            #     compress_df = compress_df.drop(columns=["id_part1", "id_part2", "id_part3", "id_part4", "id_part5"])
            # elif args.compression_method == 7:
            #     compress_df['url'] = compress_df.apply(lambda r: r["combine"][: r[f"url_len"]], axis=1)
            #     compress_df["text"] = compress_df.apply(lambda r: r["combine"][r[f"url_len"]: ], axis=1)
            #     compress_df = compress_df.drop(columns=["combine", "url_len"])
            # elif args.compression_method == 8:
            #     compress_df["url"] = compress_df.apply(lambda r: r["combine"][: r["url_len"]], axis=1)
            #     compress_df["url"] = compress_df.apply(
            #         lambda r: r["url_prefix"] if r["url"] == "" else r["url_prefix"] + "/" + r["url"],
            #         axis=1
            #     )
            #     compress_df["text"] = compress_df.apply(lambda r: r["combine"][r["url_len"]:], axis=1)
            #     compress_df = compress_df.drop(columns=["combine", "url_len", "url_prefix"])

            # compare_dataframes(df, compress_df)
            
            # Update the configuration file
            with open(configs_file, "w") as f:
                json.dump(config_dict, f, indent=2)

if __name__ == '__main__':
    main()
