
import duckdb
from urllib.parse import quote, unquote
import pandas as pd
import time
import json
import os
import argparse

def myquote(text):
    return quote(text)

def strip_and_unquote(url, prefix):
    return unquote(url.replace(prefix, ""))

def main():
    """
    This script measures the query time for different compression methods
    applied to the Wikipedia dataset.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Measure query time for Wikipedia dataset.')
    parser.add_argument('compression_method', type=int, choices=[1, 2, 3, 4], help='Compression method to use (1-4)')
    args = parser.parse_args()

    # --- Configuration and Setup ---
    dataset_id = "wikimedia/wikipedia"
    configs_file = "config.json"
    con = duckdb.connect()
    con.create_function('quote', myquote, [str], return_type=str)
    con.create_function('strip_and_unquote', strip_and_unquote, [str, str], return_type=str)

    # Initialize dictionaries to store total time
    total_time_ori = 0
    total_time_compress = 0

    config_dict = json.load(open(configs_file, "r"))

    # Define prefixes for different Wikipedia languages
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

    # --- Data Processing Loop ---
    for config in config_dict[dataset_id]["configs"]:
        print(f"  Config: {config['file']}")
        config_name = config["file"]
        if config_name == "":
            config_name = dataset_id.replace("/", "_").replace("-", "_")

        # Get the compression configuration for the current method
        compress_config = config["compress"][args.compression_method - 1]
        compress_config["query"] = {}
        compress_config["query_time"] = {}

        # Helper function to run queries, measure time, and compare results
        def _run_and_compare_query(original_df_path, compressed_df_path, ori_query, compress_query, query_name, is_multi_query=False):
            nonlocal total_time_ori, total_time_compress

            # Original query
            df_ori = pd.read_parquet(original_df_path)
            con.register("df", df_ori)
            start_time = time.time()
            result_ori = con.execute(ori_query).df()
            end_time = time.time()
            print(f"Time taken (original {query_name}): {end_time - start_time} seconds")
            total_time_ori += (end_time - start_time)
            compress_config["query_time"][f"{query_name}_ori"] = (end_time - start_time)

            # Compressed query
            df_com = pd.read_parquet(compressed_df_path)
            con.register("df", df_com)
            start_time = time.time()
            result_com = con.execute(compress_query).df()
            end_time = time.time()
            print(f"Time taken (compressed {query_name}): {end_time - start_time} seconds")
            total_time_compress += (end_time - start_time)
            compress_config["query_time"][f"{query_name}_com"] = (end_time - start_time)

            if not is_multi_query:
                if result_ori.equals(result_com):
                    print("Results are equal")
                else:
                    print("Results are not equal")
            return result_ori, result_com

        # --- Method 1: URL compression ---
        if args.compression_method == 1:
            ori_query = f"""SELECT url FROM df"""
            compress_query = f"""SELECT '{prefixs[config_name]}' || quote(title) AS url FROM df"""
            compress_config["query"]["url_ori"] = ori_query
            compress_config["query"]["url_com"] = compress_query
            
            _run_and_compare_query(
                f"datasets_parquet/{dataset_id}/{config_name}.parquet",
                f"datasets_compress/{dataset_id}/{config_name}.parquet",
                ori_query,
                compress_query,
                "url"
            )

        # --- Method 2: Title and Text combined ---
        elif args.compression_method == 2:
            ori_query_title = f"""SELECT title FROM df"""
            ori_query_text = f"""SELECT text FROM df"""
            compress_query_title = f"""SELECT SUBSTR(combine_title_text, 1, title_len) AS title FROM df"""
            compress_query_text = f"""SELECT SUBSTR(combine_title_text, title_len + 1) AS text FROM df"""
            compress_config["query"]["title_ori"] = ori_query_title
            compress_config["query"]["title_com"] = compress_query_title
            compress_config["query"]["text_ori"] = ori_query_text
            compress_config["query"]["text_com"] = compress_query_text

            result_ori_title, result_com_title = _run_and_compare_query(
                f"datasets_parquet/{dataset_id}/{config_name}.parquet",
                f"datasets_compress/{dataset_id}/{config_name}.parquet",
                ori_query_title,
                compress_query_title,
                "title",
                is_multi_query=True
            )
            result_ori_text, result_com_text = _run_and_compare_query(
                f"datasets_parquet/{dataset_id}/{config_name}.parquet",
                f"datasets_compress/{dataset_id}/{config_name}.parquet",
                ori_query_text,
                compress_query_text,
                "text",
                is_multi_query=True
            )
            if result_ori_title.equals(result_com_title) and result_ori_text.equals(result_com_text):
                print("Results are equal")
            else:
                print("Results are not equal")

        # --- Method 3: URL, Title, and Text combined ---
        elif args.compression_method == 3:
            ori_query_title = f"""SELECT title FROM df"""
            ori_query_text = f"""SELECT text FROM df"""
            ori_query_url = f"""SELECT url FROM df"""
            compress_query_title = f"""SELECT SUBSTR(combine_title_text, 1, title_len) AS title FROM df"""
            compress_query_text = f"""SELECT SUBSTR(combine_title_text, title_len + 1) AS text FROM df"""
            compress_query_url = f"""SELECT '{prefixs[config_name]}' || quote(SUBSTR(combine_title_text, 1, title_len)) AS url FROM df"""
            compress_config["query"]["title_ori"] = ori_query_title
            compress_config["query"]["title_com"] = compress_query_title
            compress_config["query"]["text_ori"] = ori_query_text
            compress_config["query"]["text_com"] = compress_query_text
            compress_config["query"]["url_ori"] = ori_query_url
            compress_config["query"]["url_com"] = compress_query_url

            result_ori_title, result_com_title = _run_and_compare_query(
                f"datasets_parquet/{dataset_id}/{config_name}.parquet",
                f"datasets_compress/{dataset_id}/{config_name}.parquet",
                ori_query_title,
                compress_query_title,
                "title",
                is_multi_query=True
            )
            result_ori_text, result_com_text = _run_and_compare_query(
                f"datasets_parquet/{dataset_id}/{config_name}.parquet",
                f"datasets_compress/{dataset_id}/{config_name}.parquet",
                ori_query_text,
                compress_query_text,
                "text",
                is_multi_query=True
            )
            result_ori_url, result_com_url = _run_and_compare_query(
                f"datasets_parquet/{dataset_id}/{config_name}.parquet",
                f"datasets_compress/{dataset_id}/{config_name}.parquet",
                ori_query_url,
                compress_query_url,
                "url",
                is_multi_query=True
            )

            if result_ori_title.equals(result_com_title) and result_ori_text.equals(result_com_text) and result_ori_url.equals(result_com_url):
                print("Results are equal")
            else:
                print("Results are not equal")

        # --- Method 4: Title from URL prefix ---
        elif args.compression_method == 4:
            ori_query_title = f"""SELECT title FROM df"""
            compress_query_title = f"""SELECT strip_and_unquote(url, '{prefixs[config_name]}') AS title FROM df"""
            compress_config["query"]["title_ori"] = ori_query_title
            compress_config["query"]["title_com"] = compress_query_title

            _run_and_compare_query(
                f"datasets_parquet/{dataset_id}/{config_name}.parquet",
                f"datasets_compress/{dataset_id}/{config_name}.parquet",
                ori_query_title,
                compress_query_title,
                "title"
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
