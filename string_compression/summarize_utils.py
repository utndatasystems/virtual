import duckdb
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.gridspec as gridspec
import matplotlib
import os
# import pandas as pd
import numpy as np

# Matplotlib configuration for LaTeX rendering
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
        Initializes the TargetColumnSQL with the target column name and virtual SQL expressions.

        Args:
            target_column: The column to summarize in the queries.
            virtual_sql: The virtual SQL expression for the target column.
            virtual_length_trick: The virtual SQL expression for the length of the target column (optional).
        """
        self.target_column = target_column
        # SQL expressions for the target column
        self.target_column_sql = f"{self.target_column}"
        self.target_column_sql_length = f"LENGTH({self.target_column})"
        # Virtual SQL expressions for the target column
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
            print_flag: If True, prints SQL queries and their results.
        """
        self.original_path = original_path
        self.compressed_path = compressed_path
        # Get file sizes
        self.original_size = os.path.getsize(original_path)
        self.compressed_size = os.path.getsize(compressed_path)
        # Connect to DuckDB
        self.con = duckdb.connect()
        self.dataset_name = dataset_name
        self.print_flag = print_flag
        self.list_of_targets_columns = []
        self.dict_of_target_columns = {}

        # Initialize the DuckDB views for the datasets
        self.register_views()
        # Get column names from the original table
        columns = self.con.execute("PRAGMA table_info('original')").fetchdf()
        self.columns_name = columns['name'].tolist()

    def register_views(self):
        """
        Registers the original and compressed datasets as views in DuckDB.
        This allows for SQL queries to be run against these datasets.
        """
        # Create a view for the original dataset
        self.con.execute(f"CREATE OR REPLACE VIEW original AS SELECT * FROM parquet_scan('{self.original_path}')")
        # Create a view for the compressed (virtual) dataset
        self.con.execute(f"CREATE OR REPLACE VIEW virtual AS SELECT * FROM parquet_scan('{self.compressed_path}')")

    def time_query(self, sql: str) -> float:
        """
        Executes a DuckDB SQL query and measures the execution time.

        Args:
            sql: The SQL query string.

        Returns:
            The time taken to execute the query in seconds.
        """
        if self.print_flag:
            print(f"Executing SQL: {sql}")
        # Start timing
        start = time.time()
        result = self.con.execute(sql).fetchdf()
        if self.print_flag:
            print(result)
        # Return elapsed time
        return time.time() - start
    
    def add_target_column(self, target_column: str, virtual_sql: str, virtual_sql_length: str = None):
        """
        Adds a target column to the list of target columns for summarization.

        Args:
            target_column: The column to summarize in the queries.
            virtual_sql: The virtual SQL expression for the target column.
            virtual_sql_length: The virtual SQL expression for the length of the target column (optional).
        """
        # Check if the target column is already in the list
        if target_column in self.list_of_targets_columns:
            print(f"Target column {target_column} already exists.")
            return
        
        # Create a TargetColumnSQL instance and add it to the dictionary and list
        target_sql = TargetColumnSQL(target_column, virtual_sql, virtual_sql_length)
        self.list_of_targets_columns.append(target_column)
        self.dict_of_target_columns[target_column] = target_sql

        print(f"Added target column: {target_column}")
    
    def query_rewrite(self, query: str, target_columns: list[str], length_sql=False, length_trick=False) -> str:
        """
        Rewrites the SQL query to use virtual SQL expressions for the target columns.

        Args:
            query: The SQL query to rewrite.
            target_columns: The columns to summarize in the queries.
            length_sql: If True, rewrites the query for length operations.
            length_trick: If True, uses the length trick for virtual SQL.

        Returns:
            The rewritten SQL query string.
        """
        # Iterate through the target columns and replace with virtual SQL expressions
        for col in target_columns:
            if col in self.dict_of_target_columns:
                target_sql = self.dict_of_target_columns[col]
                if length_sql:
                    if length_trick and target_sql.virtual_length_trick:
                        # Replace with the length trick expression
                        query = query.replace(f'LENGTH({col})', f'{target_sql.virtual_length_trick}')
                    else:
                        # Replace with the standard virtual length expression
                        query = query.replace(f'LENGTH({col})', f'{target_sql.virtual_sql_length}')
                else:
                    # Replace with the standard virtual column expression
                    query = query.replace(f' {col}', f' {target_sql.virtual_sql}')
        return query

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

        # --- Define SQL queries for summarization on the target column ---
        summarize_col_sql = f"SUMMARIZE SELECT {target_column} FROM parquet_scan('{self.original_path}')"
        summarize_col_sql_virtual = self.query_rewrite(
            f"SUMMARIZE SELECT {target_column} FROM parquet_scan('{self.compressed_path}')",
            [target_column]
        )

        # --- Define SQL queries for summarization on the LENGTH of the target column ---
        summarize_len_sql = f"SUMMARIZE SELECT LENGTH({target_column}) FROM parquet_scan('{self.original_path}')"
        summarize_len_sql_virtual = self.query_rewrite(
            f"SUMMARIZE SELECT LENGTH({target_column}) FROM parquet_scan('{self.compressed_path}')",
            [target_column],
            length_sql=True
        )
        summarize_len_sql_virtual_trick = None
        if target_sql.virtual_length_trick is not None:
            summarize_len_sql_virtual_trick = self.query_rewrite(
                f"SUMMARIZE SELECT LENGTH({target_column}) FROM parquet_scan('{self.compressed_path}')",
                [target_column],
                length_sql=True,
                length_trick=True
            )

        # --- Define SQL queries for summarization on all columns ---
        sql_all = f"SUMMARIZE SELECT {', '.join(self.columns_name)} FROM parquet_scan('{self.original_path}')"
        sql_all_virtual = self.query_rewrite(
            f"SUMMARIZE SELECT {', '.join(self.columns_name)} FROM parquet_scan('{self.compressed_path}')",
            self.list_of_targets_columns
        )
        length_expressions = [f'LENGTH({col})' for col in self.columns_name]
        sql_all_length = f"SUMMARIZE SELECT {', '.join(length_expressions)} FROM parquet_scan('{self.original_path}')"
        sql_all_length_virtual = self.query_rewrite(
            f"SUMMARIZE SELECT {', '.join(length_expressions)} FROM parquet_scan('{self.compressed_path}')",
            self.list_of_targets_columns,
            length_sql=True
        )
        sql_all_length_trick = None
        if any(self.dict_of_target_columns[col].virtual_length_trick for col in self.list_of_targets_columns if col in self.dict_of_target_columns):
            sql_all_length_trick = self.query_rewrite(
                f"SUMMARIZE SELECT {', '.join(length_expressions)} FROM parquet_scan('{self.compressed_path}')",
                self.list_of_targets_columns,
                length_sql=True,
                length_trick=True
            )

        # --- Measure query times ---
        t1 = self.time_query(summarize_col_sql)
        t2 = self.time_query(summarize_col_sql_virtual)
        t3 = self.time_query(summarize_len_sql)
        t4 = self.time_query(summarize_len_sql_virtual)
        t5 = self.time_query(summarize_len_sql_virtual_trick) if summarize_len_sql_virtual_trick else None

        t6 = self.time_query(sql_all)
        t7 = self.time_query(sql_all_virtual)
        t8 = self.time_query(sql_all_length)
        t9 = self.time_query(sql_all_length_virtual)
        t10 = self.time_query(sql_all_length_trick) if sql_all_length_trick else None
        
        # --- Prepare query times for plotting ---
        query_col_times = [t1, t2, t3, t4]
        if t5 is not None:
            query_col_times.append(t5)

        query_all_times = [t6, t7, t8, t9]
        if t10 is not None:
            query_all_times.append(t10)
            
        # Plot the results
        self.plot_results(
            query_col_times,
            query_all_times,
            target_column=target_column
        )
    
    def plot_results(self, query_col_times: list, query_all_times: list, target_column: str = "Target Column"):
        """
        Generates and displays a plot comparing file sizes and query times.

        Args:
            query_col_times: A list of query times for the target column.
            query_all_times: A list of query times for all columns.
            target_column: Name of the target column for plot titles.
        """

        # --- Calculate size and time ratios ---
        original_ratio = 100.0
        compressed_ratio = (self.compressed_size / self.original_size) * 100
        times = query_col_times
        all_times = query_all_times
        monthly_costs = [34.73, 34.73 * (self.compressed_size / self.original_size)]  # Example monthly costs in USD

        # --- Prepare data for plotting ---
        # Lists contain percentage values relative to the original
        original_list = [original_ratio, 100, 100, 100, 100]
        compressed_list = [
            compressed_ratio, 
            (times[1] / times[0]) * 100 if times[0] else 0,
            (times[3] / times[2]) * 100 if times[2] else 0,
            (all_times[1] / all_times[0]) * 100 if all_times[0] else 0,
            (all_times[3] / all_times[2]) * 100 if all_times[2] else 0
        ]
        trick_list = None
        if len(times) == 5:
            trick_list = [
                0, 0, 
                (times[4] / times[2]) * 100 if times[2] else 0,
                0, 
                (all_times[4] / all_times[2]) * 100 if len(all_times) == 5 and all_times[2] else 0
            ]

        # Determine the y-axis limit for the plots
        y_max_values = compressed_list[1:]
        if trick_list:
            y_max_values.extend(trick_list)
        y_max = max([v for v in y_max_values if v is not None] + [120]) * 1.1

        # --- Plotting setup ---
        x_labels = [
            r'$\texttt{File Size}$',
            rf'$\texttt{{SUMMARIZE\ {target_column}}}$',
            rf'$\texttt{{SUMMARIZE\ LENGTH({target_column})}}$',
            rf'$\texttt{{SUMMARIZE\ *}}$',
            rf'$\texttt{{SUMMARIZE\ LENGTH(*)}}$'
        ]
        bar_width = 0.25
        plt.rcParams.update({'font.size': 10})
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5), gridspec_kw={'wspace': 0.3})
        # fig = plt.figure(figsize=(12, 5))

        # GridSpec with 4 columns: ax1 | spacer | ax2 | ax3
        # gs = gridspec.GridSpec(1, 4, width_ratios=[1, 0, 1, 1], wspace=0.25)

        # ax1 = fig.add_subplot(gs[0])
        # ax2 = fig.add_subplot(gs[2])
        # ax3 = fig.add_subplot(gs[3])
        # Set the x-ticks for the second and third plots
        x_query = np.arange(2)
        if len(times) == 5:
            x_1 = [x_query[0] - bar_width/2, x_query[-1] - bar_width]
            x_2 = [x_query[0] + bar_width/2, x_query[-1]]
            x_3 = [0, 0, x_query[-1] + bar_width]
        else:
            x_1 = x_query - bar_width/2
            x_2 = x_query + bar_width/2

        # --- Plot 1: File size comparison ---
        ax1.bar(0 - bar_width/2, original_list[0], bar_width, color='grey', label=r'$\texttt{parquet}$')
        ax1.bar(0 + bar_width/2, compressed_list[0], bar_width, color='#007FFF', label=r'$\texttt{virtual}$')

        # Annotate cost values on top of each bar
        ax1.text(0 - bar_width/2, original_list[0] / 2, rf"\${monthly_costs[0]:.2f}/month", ha='center', va='bottom', fontsize=12)
        ax1.text(0 + bar_width/2, compressed_list[0] / 2, rf"\${monthly_costs[1]:.2f}/month", ha='center', va='bottom', fontsize=12)
        # ax1.set_xticks([0])
        ax1.set_xticks([])  
        # ax1.set_xticklabels([x_labels[0]])
        ax1.set_ylabel(r'File Size [\%]')
        ax1.grid(True)
        ax1.set_axisbelow(True)
        ax1.legend(loc='upper right')

        # --- Plot 2: Query latency for target column ---
        x_query = np.arange(2)  # Two groups of bars
        
        ax2.bar(x_1, original_list[1:3], bar_width, color='grey', label=r'$\texttt{parquet}$')
        ax2.bar(x_2, compressed_list[1:3], bar_width, color='#007FFF', label=r'$\texttt{virtual}$')
        if trick_list and trick_list[2] is not None:
            ax2.bar(x_3[2], trick_list[2], bar_width, color='orange', label=r'\textit{fast} $\texttt{virtual}$')

        ax2.set_xticks(x_query)
        ax2.set_xticklabels(x_labels[1:3], fontsize=9)
        ax2.set_ylabel(r'Query Latency [\%]')
        ax2.set_ylim(0, y_max)
        # ax2.tick_params(direction='in', length=2, width=1)
        ax2.grid(True)
        ax2.set_axisbelow(True)
        ax2.legend(loc='upper right')

        # --- Plot 3: Query latency for all columns ---
        ax3.bar(x_1, original_list[3:], bar_width, color='grey', label=r'$\texttt{parquet}$')
        ax3.bar(x_2, compressed_list[3:], bar_width, color='#007FFF', label=r'$\texttt{virtual}$')
        if trick_list and trick_list[4] is not None:
            ax3.bar(x_3[2], trick_list[4], bar_width, color='orange', label=r'\textit{fast} $\texttt{virtual}$')
        
        ax3.set_xticks(x_query)
        ax3.set_xticklabels(x_labels[3:])
        ax3.set_ylabel(r'Query Latency [\%]')
        ax3.set_ylim(0, y_max)
        # ax3.tick_params(direction='in', length=2, width=1)
        ax3.grid(True)
        ax3.set_axisbelow(True)
        ax3.legend(loc='upper right')

        # --- Final plot adjustments ---
        fig.suptitle(rf'{{\Large Dataset}}: $\texttt{{{self.dataset_name.lower()}}}$', fontsize=14)
        # plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
