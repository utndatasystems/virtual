
import pandas as pd
import os
import json
import pyarrow as pa
import argparse
from utils import write_parquet, get_df, compare_dataframes, get_size

def main():
    """
    This script compresses the Fineweb URL dataset using different methods.
    The compression method is selected using a command-line argument.
    The script reads a configuration file, processes the dataset,
    applies the selected compression, and saves the result.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Compress Fineweb URL dataset.')
    parser.add_argument('compression_method', type=int, choices=[1, 2, 3], help='Compression method to use (1, 2, or 3)')
    args = parser.parse_args()

    # --- Configuration and Setup ---
    dataset_id = "nhagar/fineweb_urls"
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
                # Method 1: Drop the 'domain' column
                compress_config["function"] = "domain are the infix of the url"
                compress_df = compress_df.drop(columns=['domain'])
            elif args.compression_method == 2:
                # Method 2: Remove the domain from the URL
                compress_config["function"] = "remove the domain in url"
                compress_df['url'] = compress_df.apply(lambda x: x['url'].replace(x['domain'], ""), axis=1)
            elif args.compression_method == 3:
                # Method 3: Store domain offset and length, then drop the 'domain' column
                compress_config["function"] = "domain are the infix of the url, and store the domain offset and length"
                df['domain_offset'] = df.apply(lambda row: row['url'].find(row['domain']), axis=1)
                df['domain_length'] = df['domain'].apply(len)
                compress_df = compress_df.drop(columns=['domain'])

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
            #     # This method does not have a decompression step
            #     pass
            # elif args.compression_method == 2:
            #     # This method does not have a decompression step
            #     pass
            # elif args.compression_method == 3:
            #     # This method does not have a decompression step
            #     pass
            # compare_dataframes(df, compress_df)
            
            # Update the configuration file
            with open(configs_file, "w") as f:
                json.dump(config_dict, f, indent=2)

if __name__ == '__main__':
    main()
