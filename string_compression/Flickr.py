
import pandas as pd
import os
import json
import pyarrow as pa
import argparse
from utils import write_parquet, get_df, compare_dataframes, get_size

def main():
    """
    This script compresses the Flickr dataset using a specific method.
    The script reads a configuration file, processes the dataset,
    applies the compression, and saves the result.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Compress Flickr dataset.')
    parser.add_argument('compression_method', type=int, choices=[1], help='Compression method to use (only 1 is available)')
    args = parser.parse_args()

    # --- Configuration and Setup ---
    dataset_id = "bigdata-pw/Flickr"
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

    # --- URL Suffix Dictionary ---
    url_suffix_dict = {
        'url_sq': '_s',
        'url_q': '_q',
        'url_t': '_t',
        'url_s': '_m',
        'url_n': '_n',
        'url_w': '_w',
        'url_m': '',
        'url_z': '_z',
        'url_c': '_c',
        'url_l': '_b',
        'url_h': '_h',
        'url_k': '_k',
        'url_3k': '_3k',
        'url_4k': '_4k',
        'url_5k': '_5k',
        'url_6k': '_6k',
    }
    url_cols = ['url_sq', 'url_q', 'url_t', 'url_s', 'url_n', 'url_w', 'url_m', 'url_z', 'url_c', 'url_l']

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
        if config_name == "":
            config_name = dataset_id.replace("/", "_").replace("-", "_")
        
        # Load the dataset
        df = get_df(config["path"])

        # --- Compression and Evaluation Loop ---
        for compression_method in compression_methods:
            compress_df = df.copy()

            # --- Compression Logic ---
            if args.compression_method == 1:
                # Method 1: Store boolean flags for URL presence and a template URL
                compress_config["function"] = "the urls have the same prefix and have same suffix for same columns"
                for url_col in url_cols:
                    compress_df[f"{url_col}_null"] = compress_df[url_col].notna()
                compress_df['url_temp']= compress_df['url_sq'].apply(lambda x: x.replace("_s", "_{}"))
                compress_df = compress_df.drop(columns=url_cols)

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
            #     for url_col in url_cols:
            #         compress_df[f"{url_col}"] = compress_df.apply(lambda x: x['url_temp'].replace("_{}", url_suffix_dict[url_col]) if x[f"{url_col}_null"] else None, axis=1)
            #         compress_df = compress_df.drop(columns=[f"{url_col}_null"])
            #     compress_df = compress_df.drop(columns=['url_temp'])

            # compare_dataframes(df, compress_df)
            
            # Update the configuration file
            with open(configs_file, "w") as f:
                json.dump(config_dict, f, indent=2)

if __name__ == '__main__':
    main()
