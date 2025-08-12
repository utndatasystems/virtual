import pandas as pd
import os
import json
import argparse
from utils import (
    get_df,
    filter_sample_df,
    get_pre_suffix
)

def analyze_pre_suffix(input_data: str|pd.DataFrame) -> None:
    """
    Analyzes a dataset for prefix and suffix relationships between columns.

    Args:
        file_path (str): The path to the dataset file.
    """
    if isinstance(input_data, pd.DataFrame):
        df = input_data
    else:
        if not os.path.exists(input_data):
            print(f"Error: File not found at {input_data}")
            return
        df = get_df(input_data)
    
    if df.empty:
        print("DataFrame is empty. Exiting.")
        return

    sampled_df, string_columns = filter_sample_df(df)
    
    if not string_columns.any():
        print("No string columns found in the dataset. Exiting.")
        return

    results = get_pre_suffix(sampled_df, string_columns)
    
    # output_filename = os.path.splitext(os.path.basename(file_path))[0] + "_results.json"
    # output_path = os.path.join("./results", output_filename)

    print(f"\nResults: {results}")
    # with open(output_path, "w") as f:
    #     json.dump(results, f, indent=4)
    # print(f"Results saved to {output_path}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze prefix and suffix relationships in a dataset.")
    parser.add_argument("file_path", type=str, help="The path to the dataset file.")
    args = parser.parse_args()
    
    analyze_pre_suffix(args.file_path)
