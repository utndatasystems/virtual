
import pandas as pd
import os
import json
from urllib.parse import quote, unquote
import pyarrow as pa
import argparse
from utils import write_parquet, get_df, compare_dataframes, get_size

def main():
    """
    This script compresses the Wikipedia dataset using different methods.
    The compression method is selected using a command-line argument.
    The script reads a configuration file, processes the dataset,
    applies the selected compression, and saves the result.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Compress Wikipedia dataset.')
    parser.add_argument('compression_method', type=int, choices=[1, 2, 3, 4], help='Compression method to use (1, 2, 3, or 4)')
    args = parser.parse_args()

    # --- Configuration and Setup ---
    os.makedirs("datasets_compress/wikimedia/wikipedia", exist_ok=True)
    dataset_id = "wikimedia/wikipedia"
    configs_file = "config.json"
    compression_methods = ["snappy", "gzip", "brotli", "lz4", "zstd"]
    config_dict = json.load(open(configs_file, "r"))
    
    # --- Prefix Dictionary ---
    prefixs = {
        '20231101.zh-classical': "https://zh-classical.wikipedia.org/wiki/", 
        '20231101.ab': "https://ab.wikipedia.org/wiki/", 
        '20231101.af': "https://af.wikipedia.org/wiki/", 
        '20231101.azb': "https://azb.wikipedia.org/wiki/", 
        '20231101.bg': "https://bg.wikipedia.org/wiki/",
        '20231101.ady': "https://ady.wikipedia.org/wiki/",
        '20231101.ar': "https://ar.wikipedia.org/wiki/",
        '20231101.de': "https://de.wikipedia.org/wiki/",
        '20231101.ceb': "https://ceb.wikipedia.org/wiki/",
        '20231101.nl': "https://nl.wikipedia.org/wiki/"
    }

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
        df = get_df(f"datasets/{dataset_id}/{config_name}/")

        # --- Compression and Evaluation Loop ---
        for compression_method in compression_methods:
            compress_df = df.copy()

            # --- Compression Logic ---
            if args.compression_method == 1:
                # Method 1: Drop the 'url' column
                compress_config["function"] = "url=offset+transfer(title)"
                compress_df = df.drop(columns=["url"])
            elif args.compression_method == 2:
                # Method 2: Combine 'title' and 'text' columns
                compress_config["function"] = "title=combine[:len], text=combine[len:]"
                compress_df["combine_title_text"] = compress_df["title"] + compress_df["text"]
                compress_df["title_len"] = compress_df["title"].str.len()
                compress_df = compress_df.drop(columns=["title", "text"])
            elif args.compression_method == 3:
                # Method 3: Combination of method 1 and 2
                compress_config["function"] = "url=offset+transfer(title), title=combine[:len], text=combine[len:]"
                compress_df = df.drop(columns=["url"])
                compress_df["combine_title_text"] = compress_df["title"] + compress_df["text"]
                compress_df["title_len"] = compress_df["title"].str.len()
                compress_df = compress_df.drop(columns=["title", "text"])
            elif args.compression_method == 4:
                # Method 4: Drop the 'title' column
                compress_config["function"] = "title=transfer(url-prefix)"
                compress_df = df.drop(columns=["title"])

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
            #     compress_df["url"] = compress_df["title"].apply(lambda x: prefixs[config_name] + quote(x))
            # elif args.compression_method == 2:
            #     compress_df["title"] = compress_df.apply(lambda r: r["combine_title_text"][: r["title_len"]], axis=1, result_type="reduce")
            #     compress_df["text"] = compress_df.apply(lambda r: r["combine_title_text"][r["title_len"] :], axis=1, result_type="reduce")
            #     compress_df = compress_df.drop(columns=["combine_title_text", "title_len"])
            # elif args.compression_method == 3:
            #     compress_df["title"] = compress_df.apply(lambda r: r["combine_title_text"][: r["title_len"]], axis=1, result_type="reduce")
            #     compress_df["text"] = compress_df.apply(lambda r: r["combine_title_text"][r["title_len"] :], axis=1, result_type="reduce")
            #     compress_df["url"] = compress_df["title"].apply(lambda x: prefixs[config_name] + quote(x))
            #     compress_df = compress_df.drop(columns=["combine_title_text", "title_len"])
            # elif args.compression_method == 4:
            #     prefix = prefixs[config_name]
            #     compress_df["title"] = compress_df["url"].apply(lambda x: unquote(x.replace(prefix, "")))

            # compare_dataframes(df, compress_df)
            
            # Update the configuration file
            with open(configs_file, "w") as f:
                json.dump(config_dict, f, indent=2)

if __name__ == '__main__':
    main()
