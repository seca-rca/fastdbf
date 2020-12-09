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

```python
from fastdbf import to_df
df = to_df(filename)
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