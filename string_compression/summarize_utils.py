import duckdb
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib
import os
# import pandas as pd
import numpy as np

matplotlib.rcParams.update({
  "pgf.texsystem": "pdflatex",
  'font.family': 'serif',
  'text.usetex': True,
  'pgf.rcfonts': False,
})

class TargetColumnSQL:
    """
    A class to handle SQL queries for summarizing a target column in a dataset.
    It provides methods to generate SQL queries for summarization and length operations.
    """

    def __init__(self, target_column: str, virtual_sql: str, virtual_length_trick: str = None):
        """
        Initializes the TargetColumnSQL with the target column name.

        Args:
            target_column: The column to summarize in the queries.
        """
        self.target_column = target_column
        self.target_column_sql = f"{self.target_column}"
        self.target_column_sql_length = f"LENGTH({self.target_column})"
        self.virtual_sql = f"{virtual_sql} AS {self.target_column}"
        self.virtual_sql_length = f"LENGTH({virtual_sql}) AS 'LENGTH({self.target_column})'"
        if virtual_length_trick:
            self.virtual_length_trick = f"{virtual_length_trick} AS 'LENGTH({self.target_column})'"
        else:
            self.virtual_length_trick = None



class SummarizePlotter:
    """
    A class to handle plotting of results for string compression summarization.
    It provides methods to plot file sizes and query times for original and compressed datasets.
    """

    def __init__(self, original_path: str, compressed_path: str, dataset_name: str, print_flag:bool = False):
        """
        Initializes the SummarizePlotter with paths to original and compressed datasets.

        Args:
            original_path: Path to the original dataset file.
            compressed_path: Path to the compressed dataset file.
            dataset_name: Name of the dataset for labeling plots.
        """
        self.original_path = original_path
        self.compressed_path = compressed_path
        self.original_size = os.path.getsize(original_path)
        self.compressed_size = os.path.getsize(compressed_path)
        self.con = duckdb.connect()
        self.dataset_name = dataset_name
        self.print_flag = print_flag
        self.list_of_targets_columns = []
        self.dict_of_target_columns = {}

        # Initialize the DuckDB views for the datasets
        self.register_views()

    def register_views(self):
        """
        Registers the original and compressed datasets as views in DuckDB.
        This allows for SQL queries to be run against these datasets.
        """
        self.con.execute(f"CREATE OR REPLACE VIEW original AS SELECT * FROM parquet_scan('{self.original_path}')")
        self.con.execute(f"CREATE OR REPLACE VIEW virtual AS SELECT * FROM parquet_scan('{self.compressed_path}')")

    def time_query(self, sql: str) -> float:
        """
        Executes a DuckDB SQL query and measures the execution time.

        Args:
            sql: The SQL query string.
            print_result: If True, prints the DataFrame result of the query.

        Returns:
            The time taken to execute the query in seconds.
        """
        if self.print_flag:
            print(f"Executing SQL: {sql}")
        start = time.time()
        result = self.con.execute(sql).fetchdf()
        if self.print_flag:
            print(result)
        return time.time() - start
    
    def add_target_column(self, target_column: str, virtual_sql: str, virtual_sql_length: str = None):
        """
        Adds a target column to the list of target columns for summarization.

        Args:
            target_column: The column to summarize in the queries.
        """
        # Check if the target column is already in the list
        if target_column in self.list_of_targets_columns:
            print(f"Target column {target_column} already exists.")
            return
        
        # Create a TargetColumnSQL instance and add it to the list
        target_sql = TargetColumnSQL(target_column, virtual_sql, virtual_sql_length)
        self.list_of_targets_columns.append(target_column)
        self.dict_of_target_columns[target_column] = target_sql

        print(f"Added target column: {target_column}")
    
    def plot_target_column(self, target_column: str):
        """
        Generates and displays a plot comparing file sizes and query times for the specified target column.

        Args:
            target_column: The column to summarize in the queries.
        """
        
        # Check if the target column is already in the list
        if target_column not in self.list_of_targets_columns:
            print(f"No target column {target_column} found. Please add it first.")
            return
        
        target_sql = self.dict_of_target_columns[target_column]

        # Define SQL queries for summarization operations
        summarize_col_sql = f"SUMMARIZE SELECT {target_sql.target_column_sql} FROM original"
        summarize_col_sql_virtual = f"SUMMARIZE SELECT {target_sql.virtual_sql} FROM virtual"

        # Define SQL queries for summarization length operations
        summarize_len_sql = f"SUMMARIZE SELECT {target_sql.target_column_sql_length} FROM original"
        summarize_len_sql_virtual = f"SUMMARIZE SELECT {target_sql.virtual_sql_length} FROM virtual"
        if target_sql.virtual_length_trick is not None:
            summarize_len_sql_virtual_trick = f"SUMMARIZE SELECT {target_sql.virtual_length_trick} FROM virtual"

        # Measure query times
        t1 = self.time_query(summarize_col_sql)
        t2 = self.time_query(summarize_col_sql_virtual)
        t3 = self.time_query(summarize_len_sql)
        t4 = self.time_query(summarize_len_sql_virtual)
        if target_sql.virtual_length_trick is not None:
            t5 = self.time_query(summarize_len_sql_virtual_trick)

        
        # Print query times if print_flag is set
        if self.print_flag:
            print(f"Original query time: {t1:.4f} seconds")
            print(f"Compressed query time: {t2:.4f} seconds")
            print(f"Original query time: {t3:.4f} seconds")
            print(f"Compressed query time: {t4:.4f} seconds")
            if target_sql.virtual_length_trick is not None:
                print(f"Trick query time: {t5:.4f} seconds")
        
        # Prepare query times for plotting
        query_times = [t1, t2, t3, t4]
        if target_sql.virtual_length_trick is not None:
            query_times.append(t5)
            
        # Plot the results
        self.plot_results(
            query_times
        )
    
    def plot_results(self, query_times: list):
        """
        Generates and displays a plot comparing file sizes and query times.

        Args:
            original_size: Size of the original file in bytes.
            compressed_size: Size of the compressed file in bytes.
            query_times: A dictionary containing query labels and their corresponding times.
            dataset_name: Name of the dataset for plot titles.
        """
        plt.figure(figsize=(12, 5))

        # Left: File sizes
        original_ratio = (self.original_size / self.original_size) * 100
        compressed_ratio = (self.compressed_size / self.original_size) * 100
        # labels = list(query_times.keys())
        times = query_times

        original_list = [original_ratio, (times[0] / times[0]) * 100, (times[2] / times[2]) * 100]
        compressed_list = [compressed_ratio, (times[1] / times[0]) * 100, (times[3] / times[2]) * 100]
        if len(times) == 5:
            trick_list = [0, 0, (times[4] / times[2]) * 100]
        x_labels = ['Size', 'SUMMARIZE column', 'SUMMARIZE LENGTH(column)']

        bar_width = 0.25
        x = np.arange(len(x_labels))

        if len(times) == 5:
            x_1 = [x[i] - bar_width/2 for i in range(2)]
            x_2 = [x[i] + bar_width/2 for i in range(2)]
            x_1.append(x[-1] - bar_width)
            x_2.append(x[-1])
            x_3 = [0, 0, x[-1] + bar_width]
        else:
            x_1 = x - bar_width/2
            x_2 = x + bar_width/2

        plt.bar(x_1, original_list, bar_width, color=['blue'], label='Original')
        plt.bar(x_2, compressed_list, bar_width, color=['orange'], label='Compressed')
        if len(times) == 5:
            plt.bar(x_3, trick_list, bar_width, color=['yellow'], label='Compressed Len Trick')

        plt.ylabel('Percentage(%)')
        plt.xticks(x, x_labels)
        # log scale y
        # plt.yscale('log')
        plt.title(f'Parquet Compression {self.dataset_name}')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()
