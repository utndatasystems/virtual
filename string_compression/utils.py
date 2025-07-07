import pandas as pd
import os
import json
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Tuple, List, Dict, Any

def write_parquet(df: pd.DataFrame, path: str, PARQUET_COMPRESSION_TYPE: str = "snappy"):
    """
    Writes a Pandas DataFrame to a Parquet file with specified compression.

    Args:
        df: The DataFrame to write.
        path: The output file path.
        PARQUET_COMPRESSION_TYPE: The compression codec to use (e.g., "snappy", "gzip").
    """
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path, compression=PARQUET_COMPRESSION_TYPE)

def get_df(path: str, max_rows: int = 1_000_000) -> pd.DataFrame:
    """
    Reads multiple Parquet files from a directory into a single DataFrame.

    Stops reading once max_rows is reached.

    Args:
        path: The directory containing the Parquet files.
        max_rows: The maximum number of rows to load.

    Returns:
        A concatenated Pandas DataFrame.
    """
    parts = []
    total_rows = 0

    if not os.path.isdir(path):
        print(f"Error: Path is not a directory: {path}")
        return pd.DataFrame()

    for fname in sorted(os.listdir(path)):
        if not fname.endswith(('.parquet', '.parq')):
            continue

        file_path = os.path.join(path, fname)
        try:
            df_part = pd.read_parquet(file_path)
            n = len(df_part)

            if total_rows + n >= max_rows:
                # Only take what we still need to reach the cap
                rows_to_take = max_rows - total_rows
                df_part = df_part.iloc[:rows_to_take]
                parts.append(df_part)
                break

            parts.append(df_part)
            total_rows += n
        except Exception as e:
            print(f"Warning: Could not read {file_path}. Error: {e}")
            continue

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True)

def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    Compares two Pandas DataFrames and prints out differences.

    Args:
        df1: The first DataFrame.
        df2: The second DataFrame.
    """
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

def get_size(start_path: str = '.') -> int:
    """
    Calculates the total size of a file or a directory and its contents.

    Args:
        start_path: The path to the file or directory.

    Returns:
        The total size in bytes.
    """
    # check is it a file or directory
    if os.path.isfile(start_path):
        return os.path.getsize(start_path)
    total = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # Ensure the path is a file and not a broken symlink
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total

def filter_sample_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Index]:
    """
    Filters a DataFrame to include only string columns and samples 10,000 rows.

    Args:
        df: The input DataFrame.

    Returns:
        A tuple containing the sampled DataFrame and its string column names.
    """
    string_columns = df.select_dtypes(include=['object', 'string']).columns
    df = df[string_columns]
    sampled_df = df.sample(n=10000, random_state=42)
    return sampled_df, string_columns

def get_pre_suffix(sampled_df: pd.DataFrame, string_columns: pd.Index, filter_threshold: float = 0.9) -> List[Dict[str, Any]]:
    """
    Compares string columns in a sampled DataFrame for prefix, suffix, and infix matches.

    Args:
        sampled_df: The DataFrame containing sampled data.
        string_columns: A Pandas Index of string column names to compare.
        filter_threshold: The minimum mean match ratio to consider a match.

    Returns:
        A list of dictionaries, each containing information about column pairs
        that meet the filter threshold for various match types.
    """
    print(f"Comparing {len(string_columns)} string columns for prefix and suffix matches...")
    print(f"Filter threshold set to {filter_threshold:.2f}")
    results = []
    count = 0
    total_comparisons = len(string_columns) * (len(string_columns) - 1)

    for col1 in string_columns:
        for col2 in string_columns:
            if col1 == col2:
                continue  # skip same column
            print(f"Processing ({count+1}/{total_comparisons})", end='\r')
            count += 1
            temp = {}

            sampled_df["equal_match"] = sampled_df.apply(
                lambda x: x[col1] == x[col2] if pd.notna(x[col1]) and pd.notna(x[col2]) else False,
                axis=1
            )
            if sampled_df["equal_match"].mean() >= 1.0:
                temp['equal_match'] = sampled_df["equal_match"].mean()
            else:
                if sampled_df["equal_match"].mean() >= filter_threshold:
                    temp = {'equal_match': sampled_df["equal_match"].mean()}

                sampled_df["containment_ratio"] = sampled_df.apply(
                    lambda x: (
                        longest_common_substring(x[col2], x[col1])
                        if pd.notna(x[col1]) and pd.notna(x[col2]) and len(x[col2]) > 0 else 0
                    ),
                    axis=1
                )
                avg_ratio = sampled_df["containment_ratio"] / sampled_df[col2].str.len().replace(0, 1).fillna(1)  # Avoid division by zero
                avg_ratio = avg_ratio.mean()
                if avg_ratio >= filter_threshold:
                    temp['containment_ratio'] = avg_ratio

                if avg_ratio < filter_threshold:
                    continue

                # len of col2 need to longer than 2
                if sampled_df[col2].str.len().mean() <= 2:
                    continue

                if avg_ratio != 0:
                    sampled_df["prefix_match"] = sampled_df.apply(
                        lambda x: x[col1].startswith(x[col2][:x['containment_ratio']]) if pd.notna(x[col1]) and pd.notna(x[col2]) else False,
                        axis=1
                    )
                    sampled_df["suffix_match"] = sampled_df.apply(
                        lambda x: x[col1].endswith(x[col2]) if pd.notna(x[col1]) and pd.notna(x[col2]) else False,
                        axis=1
                    )
                else:
                    sampled_df["prefix_match"] = sampled_df.apply(
                        lambda x: x[col1].startswith(x[col2]) if pd.notna(x[col1]) and pd.notna(x[col2]) else False,
                        axis=1
                    )
                    sampled_df["suffix_match"] = sampled_df.apply(
                        lambda x: x[col1].endswith(x[col2]) if pd.notna(x[col1]) and pd.notna(x[col2]) else False,
                        axis=1
                    )
                if sampled_df["prefix_match"].mean() >= filter_threshold:
                    temp['prefix_match'] = sampled_df["prefix_match"].mean()
                elif sampled_df["suffix_match"].mean() >= filter_threshold:
                    temp['suffix_match'] = sampled_df["suffix_match"].mean()
                else:
                    sampled_df["infix_match"] = sampled_df.apply(
                        lambda x: x[col2] in x[col1] if pd.notna(x[col1]) and pd.notna(x[col2]) else False,
                        axis=1
                    )
                    
                    if sampled_df["infix_match"].mean() >= filter_threshold:
                        temp['infix_match'] = sampled_df["infix_match"].mean()

                sampled_df.drop(columns=["prefix_match", "suffix_match"], inplace=True)
            
            if temp == {}:
                continue
            results.append({
                'col1': col1,
                'col2': col2,
                **temp
            })
            
    
    return results

def longest_common_substring(a: str, b: str) -> int:
    """
    Calculates the length of the longest common substring between two strings.

    Args:
        a: The first string.
        b: The second string.

    Returns:
        The length of the longest common substring.
    """
    if not a or not b:
        return 0
    dp = [[0] * (len(b)+1) for _ in range(len(a)+1)]
    max_len = 0
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                max_len = max(max_len, dp[i][j])
    return max_len