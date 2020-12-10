import unittest
import fastdbf
import os
import pandas as pd
import numpy as np
import tempfile
import shutil

class TestToDF(unittest.TestCase):
    def setUp(self):
        self.file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'testdata', 'test.dbf')
        self.multi_path = tempfile.mkdtemp()
        self.multi_dirs = [os.path.join(self.multi_path, f"YEAR-{year}") for year in [2015, 2016, 2017, 2018, 2019, 2020]]
        
        for i, multi_dir in enumerate(self.multi_dirs):
            os.makedirs(multi_dir, exist_ok=True)
            shutil.copy(self.file_path, os.path.join(multi_dir, 'TEST.DBF' if i % 2 == 0 else 'test.dbf'))
    
    def tearDown(self):
        shutil.rmtree(self.multi_path, ignore_errors=True)

    def test_multi_df(self):
        df = fastdbf.multi_df(self.multi_dirs, os.path.basename(self.file_path))
        self.assertEqual(len(df), len(self.multi_dirs) * 2)
        self.assertEqual(df.columns.to_list(), ['int_field', 'flt_field', 'bool_field', 'num_field', 'date_field', 'char_field'])
        self.assertEqual(df.int_field.to_list(), [123, 456] * len(self.multi_dirs))
        np.testing.assert_almost_equal(df.flt_field, [123.456, 456.789] * len(self.multi_dirs), 3)
        np.testing.assert_almost_equal(df.num_field, [123.456, 456.789] * len(self.multi_dirs), 3)
        self.assertEqual(df.bool_field.to_list(), [True, False] * len(self.multi_dirs))
        self.assertEqual(df.char_field.to_list(), ['This is it', 'Döt Üré'] * len(self.multi_dirs))
        self.assertEqual(df.date_field.to_list(), [pd.Timestamp('2020-12-10 00:00:00'), pd.Timestamp('2020-12-09 00:00:00')] * len(self.multi_dirs))

    def test_to_df(self):
        df = fastdbf.to_df(self.file_path)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.columns.to_list(), ['int_field', 'flt_field', 'bool_field', 'num_field', 'date_field', 'char_field'])
        self.assertAlmostEqual(df.flt_field[0], 123.456, 3)
        self.assertAlmostEqual(df.flt_field[1], 456.789, 3)
        self.assertAlmostEqual(df.num_field[0], 123.456, 3)
        self.assertAlmostEqual(df.num_field[1], 456.789, 3)
        self.assertEqual(df.int_field[0], 123)
        self.assertEqual(df.int_field[1], 456)
        self.assertEqual(df.bool_field[0], True)
        self.assertEqual(df.bool_field[1], False)
        self.assertEqual(df.date_field[0], pd.Timestamp('2020-12-10 00:00:00'))
        self.assertEqual(df.date_field[1], pd.Timestamp('2020-12-09 00:00:00'))
        self.assertEqual(df.char_field[0], 'This is it')
        self.assertEqual(df.char_field[1], 'Döt Üré')


if __name__ == '__main__':
    unittest.main()