# `virtual`

A booster 💪 for your Parquet file sizes.

# 🛠 Build

```
pip3 install virtual-parquet
```

or 

```
pip3 install .
```

# 🔗 Examples

A demo can be found at `examples/demo.ipynb`.

## 🗜️ Compress

```python
import pandas as pd
import virtual

df = pd.read_csv('file.csv')

...

virtual.to_parquet(df, 'file_virtual.parquet')
```
> % Virtualization finished: Check out 'file.parquet'.

## 🥢 Read

```python
import virtual

df = virtual.from_parquet('file_virtual.parquet')
```

## 📊 Query

```python
import virtual

virtual.query(
  'select avg(price) from read_parquet("file_virtual.parquet") where year >= 2024',
  engine = 'duckdb'
)
```

# Additional Features

## 🔍 Discover the Functions Found

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