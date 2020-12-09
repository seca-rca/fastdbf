import fastdbf as f
import os

f.to_df(os.path.join(os.path.dirname(__file__), "testdata", "faktp.DBF"))