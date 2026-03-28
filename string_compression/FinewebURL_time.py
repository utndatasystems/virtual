import duckdb
import pandas as pd
import time
import json
import os
import argparse

def main():
    """
    This script measures the query time for different compression methods
    applied to the Fineweb URL dataset.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Measure query time for Fineweb URL dataset.')
    parser.add_argument('compression_method', type=int, choices=[1, 3], help='Compression method to use (1 or 3)')
    args = parser.parse_args()

    # --- Configuration and Setup ---
    dataset_id = "nhagar/fineweb_urls"
    configs_file = "config.json"
    con = duckdb.connect()

    # Initialize dictionaries to store total time
    total_time_ori = 0
    total_time_compress = 0

    config_dict = json.load(open(configs_file, "r"))

    # --- Data Processing Loop ---
    for config in config_dict[dataset_id]["configs"]:
        print(f"  Config: {config['file']}")
        config_name = config["file"]
        if config_name == "":
            config_name = dataset_id.replace("/", "_").replace("-", "_")

        # Get the compression configuration for the current method
        if args.compression_method == 1:
            compression_method_index = 0
        elif args.compression_method == 3:
            compression_method_index = 2
        else:
            raise ValueError("Invalid compression method selected.")

        compress_config = config["compress"][compression_method_index]
        compress_config["query"] = {}
        compress_config["query_time"] = {}

        # Helper function to run queries, measure time, and compare results
        def _run_and_compare_query(original_df_path, compressed_df_path, ori_query, compress_query, query_name):
            nonlocal total_time_ori, total_time_compress

            # Original query
            df_ori = pd.read_parquet(original_df_path)
            con.register("df", df_ori)
            start_time = time.time()
            result_ori = con.execute(ori_query).df()
            print(f"Original result: {result_ori.head()}")
            end_time = time.time()
            print(f"Time taken (original {query_name}): {end_time - start_time} seconds")
            total_time_ori += (end_time - start_time)
            compress_config["query_time"][f"{query_name}_ori"] = (end_time - start_time)

            # Compressed query
            df_com = pd.read_parquet(compressed_df_path)
            con.register("df", df_com)
            start_time = time.time()
            result_com = con.execute(compress_query).df()
            print(f"Compressed result: {result_com.head()}")
            end_time = time.time()
            print(f"Time taken (compressed {query_name}): {end_time - start_time} seconds")
            total_time_compress += (end_time - start_time)
            compress_config["query_time"][f"{query_name}_com"] = (end_time - start_time)

            if result_ori.equals(result_com):
                print("Results are equal")
            else:
                # compare the two DataFrames
                diff = result_ori.compare(result_com)
                print("Differences found:")
                print(diff)
                print("Results are not equal")

        # --- Method 1: URL domain extraction using regex ---
        if args.compression_method == 1:
            ori_query = f"""SELECT domain FROM df"""
            compress_query = f"""SELECT regexp_extract(url, 'https?://(?:[^./]+\\.)*([^./]+\\.[^./]+)', 1) AS domain FROM df"""
            compress_config["query"]["url_ori"] = ori_query
            compress_config["query"]["url_com"] = compress_query
            
            _run_and_compare_query(
                f"datasets_parquet/{dataset_id}/{config_name}.parquet",
                f"datasets_compress/{dataset_id}/{config_name}.parquet",
                ori_query,
                compress_query,
                "url"
            )

        # --- Method 3: URL domain extraction using substring ---
        elif args.compression_method == 3:
            ori_query = f"""SELECT domain FROM df"""
            compress_query = f"""SELECT substr(url, domain_offset + 1, domain_length) AS domain FROM df"""
            compress_config["query"]["url_ori"] = ori_query
            compress_config["query"]["url_com"] = compress_query

            _run_and_compare_query(
                f"datasets_parquet/{dataset_id}/{config_name}.parquet",
                f"datasets_compress/{dataset_id}/{config_name}.parquet",
                ori_query,
                compress_query,
                "url"
            )

        # --- Update Config File ---
        with open(configs_file, "w") as f:
            json.dump(config_dict, f, indent=2)

    # --- Final Time Summary ---
    print(f"Total time taken (original): {total_time_ori} seconds")
    print(f"Total time taken (compress): {total_time_compress} seconds")
    con.close()

if __name__ == '__main__':
    main()
