import pandas as pd
import os
import json
import pyarrow as pa
import argparse
from utils import write_parquet, get_df, compare_dataframes, get_size

def main():
    """
    This script compresses the Metanova dataset using different methods.
    The compression method is selected using a command-line argument.
    The script reads a configuration file, processes the dataset,
    applies the selected compression, and saves the result.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Compress Metanova dataset.')
    parser.add_argument('compression_method', type=int, choices=range(1, 6), help='Compression method to use (1-5)')
    args = parser.parse_args()

    # --- Configuration and Setup ---
    dataset_id = "Metanova/SAVI-2020"
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
        
        # Load the dataset
        df = get_df(config["path"])

        # --- Compression and Evaluation Loop ---
        for compression_method in compression_methods:
            compress_df = df.copy()

            # --- Compression Logic ---
            if args.compression_method == 1:
                # Method 1: Drop the 'product_hashisy' column
                compress_config["function"] = "remove product_hashisy, product_hashisy=product_name[:19]+'"'"
                compress_df = compress_df.drop(columns=['product_hashisy'], axis=1)
            elif args.compression_method == 2:
                # Method 2: Remove the product hash from the product name
                compress_config["function"] = "product_name = product_name[0] + product_hashisy[1:-1] + product_name[1:]"
                compress_df['product_name'] = compress_df.apply(lambda x: x['product_name'].replace(x['product_hashisy'][1:-1], ""), axis=1)
            elif args.compression_method == 3:
                # Method 3: Drop the 'r1_ident' and 'r2_ident' columns
                compress_config["function"] = "remove r1_ident, and r1_ident=r1_url.split('/')[-1], same for r2"
                compress_df = compress_df.drop(columns=['r1_ident','r2_ident'], axis=1)
            elif args.compression_method == 4:
                # Method 4: Remove the ident from the URL
                compress_config["function"] = "r1_url = r1_url[:-1] + r1_ident[1:-1] + r1_url[-1], same for r2"
                compress_df['r1_url'] = compress_df.apply(lambda x: x['r1_url'].replace(x['r1_ident'][1:-1], ""), axis=1)
                compress_df['r2_url'] = compress_df.apply(lambda x: x['r2_url'].replace(x['r2_ident'][1:-1], ""), axis=1)
            elif args.compression_method == 5:
                # Method 5: Combination of method 1 and 4
                compress_config["function"] = "function 1 + 4"
                compress_df = compress_df.drop(columns=['product_hashisy'], axis=1)
                compress_df['r1_url'] = compress_df.apply(lambda x: x['r1_url'].replace(x['r1_ident'][1:-1], ""), axis=1)
                compress_df['r2_url'] = compress_df.apply(lambda x: x['r2_url'].replace(x['r2_ident'][1:-1], ""), axis=1)

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
            #     compress_df['product_hashisy'] = compress_df['product_name'].apply(lambda x: x.split('_')[0] + '"')
            # elif args.compression_method == 2:
            #     compress_df['product_name'] = compress_df['product_name'].str[0] + compress_df['product_hashisy'].str[1:-1] + compress_df['product_name'].str[1:]
            # elif args.compression_method == 3:
            #     compress_df['r2_ident'] = compress_df['r2_url'].apply(lambda x: '"' + x.split("/")[-1])
            #     compress_df['r1_ident'] = compress_df['r1_url'].apply(lambda x: '"' + x.split("/")[-1])
            # elif args.compression_method == 4:
            #     compress_df['r1_url'] = compress_df['r1_url'].str[:-1] + compress_df['r1_ident'].str[1:-1] + compress_df['r1_url'].str[-1]
            #     compress_df['r2_url'] = compress_df['r2_url'].str[:-1] + compress_df['r2_ident'].str[1:-1] + compress_df['r2_url'].str[-1]
            # elif args.compression_method == 5:
            #     compress_df['product_hashisy'] = compress_df['product_name'].apply(lambda x: x.split('_')[0] + '"')
            #     compress_df['r1_url'] = compress_df['r1_url'].str[:-1] + compress_df['r1_ident'].str[1:-1] + compress_df['r1_url'].str[-1]
            #     compress_df['r2_url'] = compress_df['r2_url'].str[:-1] + compress_df['r2_ident'].str[1:-1] + compress_df['r2_url'].str[-1]

            # compare_dataframes(df, compress_df)
            
            # Update the configuration file
            with open(configs_file, "w") as f:
                json.dump(config_dict, f, indent=2)


if __name__ == '__main__':
    main()
