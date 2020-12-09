import time
import logging
import pandas as pd
import os
import argparse
import sys

logging.basicConfig(level=logging.INFO)

def test_dbfread(filename):
    from dbfread import DBF
    logging.info("Benchmark -- dbfread-DataFrame: started")
    start = time.perf_counter()
    df = pd.DataFrame(iter(DBF(filename, ignore_missing_memofile=True)))
    end = time.perf_counter()
    logging.info("Benchmark -- dbfread-DataFrame: %.0f ms", (end - start) * 1000.0)
    logging.info(df.info())

def test_fastdbf(filename):
    from fastdbf import to_df
    logging.info("Benchmark -- fastdbf-DataFrame: started")
    start = time.perf_counter()
    df = to_df(filename)
    end = time.perf_counter()
    logging.info("Benchmark -- fastdbf-DataFrame: %.0f ms", (end - start) * 1000.0)
    logging.info(df.info())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmarkt DBF readers.')
    parser.add_argument('adapter', help='which DBF reader to use', choices=['dbfread', 'fastdbf'])
    parser.add_argument('--filename', help='DBF file to use', default=os.path.join(os.path.dirname(__file__), "testdata", "pohyby.DBF"))
    
    args = parser.parse_args()

    locals()["test_" + args.adapter](args.filename)