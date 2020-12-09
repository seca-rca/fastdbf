# fastdbf

Read DBF Files with C and Python, fast.

Uses a C extension when available, otherwise pure Python.

## Installation

### Required libraries installation

Ubuntu / Debian:

```sh
# apt install libicu-dev
```

### PyPi package

```sh
$ pip install fastdbf
```

## Usage

Load a single DBF file using `to_df`:

```python
from fastdbf import to_df
df = to_df(filename)
```

Load multiple DBF files using `multi_df`;
this function will look for the specified file name case-insensitive in all directories.
the files will be loaded concurrently using `to_df` and `concurrent.futures.ProcessPoolExecutor`;
the resulting DataFrames are merged using `pandas.concat`.

```python
from fastdbf import multi_df
df = multi_df([f"/home/johndoe/mydata/DATA{year}.12" for year in (2017, 2018, 2019, 2020)], "inventory.DBF")
```

## Benchmarks

~40.0 MB DBF file, 40 columns of mixed integer, float, date, text data.

[dbfread](https://github.com/olemb/dbfread/):

```sh
INFO:root:Benchmark -- dbfread-DataFrame: started
INFO:root:Benchmark -- dbfread-DataFrame: 13458 ms
```

fastdbf:

```sh
INFO:root:Benchmark -- fastdbf-DataFrame: started
INFO:root:Benchmark -- fastdbf-DataFrame: 1010 ms
```