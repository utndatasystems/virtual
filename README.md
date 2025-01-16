# `virtual`

A booster 💪 for your Parquet file sizes.

`virtual` is a lightweight framework that transparently compresses Parquet files by using functions between columns, all while giving you the same familiar interface you are used to. How `virtual` works is magic, and is described in our recent research papers (see below).

# 🛠 Build

```
pip install virtual-parquet
```

or 

```
git clone https://github.com/utndatasystems/virtual.git && cd virtual
pip install .
```

# 🔗 Examples

A demo can be found at [`examples/demo-parquet.ipynb`](examples/demo-parquet.ipynb).

## 🗜️ Compress

Simply compress a Pandas DataFrame with `virtual.to_format(df)`:

```python
import pandas as pd
import virtual

df = pd.read_csv('file.csv')

...

virtual.to_format(df, 'file_virtual.parquet')
```
> % Virtualization finished: Check out 'file_virtual.parquet'.

## 🥢 Read

Reading in a virtual compress parquet file with `virtual.from_format([path])`:

```python
import virtual

df = virtual.from_format('file_virtual.parquet')
```

## 📊 Query

Or directly run SQL queries on the virtualized Parquet file via [duckdb](https://github.com/duckdb/duckdb) with `virtual.query([SQL])`:

```python
import virtual

virtual.query(
  'select avg(price) from read_parquet("file_virtual.parquet") where year >= 2024',
  engine = 'duckdb'
)
```

# Expert-User Features

## 🔍 Inspect the Functions Found

```python
import pandas as pd
import virtual

df = pd.read_csv('file.csv')

functions = virtual.train(df)
```
> % Functions saved under `functions.json`.


# 📚 Citation

Please do cite our (very) cool work if you use `virtual` in your work.

```
@inproceedings{
  virtual,
  title={{Lightweight Correlation-Aware Table Compression}},
  author={Mihail Stoian and Alexander van Renen and Jan Kobiolka and Ping-Lin Kuo and Josif Grabocka and Andreas Kipf},
  booktitle={NeurIPS 2024 Third Table Representation Learning Workshop},
  year={2024},
  url={https://openreview.net/forum?id=z7eIn3aShi}
}
```