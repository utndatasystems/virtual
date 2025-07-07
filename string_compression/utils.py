import pandas as pd
import os
import json
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

def compare_dataframes(df1, df2):
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

def filter_sample_df(df):

    string_columns = df.select_dtypes(include=['object', 'string']).columns
    df = df[string_columns]
    sampled_df = df.sample(n=10000, random_state=42)
    return sampled_df, string_columns

def get_pre_suffix(sampled_df, string_columns, filter_threshold=0.9):
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

def longest_common_substring(a, b):
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