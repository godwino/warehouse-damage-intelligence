"""Exercise the real Streamlit cache, including the failing pandas hash path."""
import unittest
from unittest.mock import patch
import pandas as pd
import app
from generate_data import generate_dataset

class ModelCacheTests(unittest.TestCase):
    def tearDown(self):
        app.train_model.clear()

    def test_cache_works_when_default_pandas_hashing_fails(self):
        data = generate_dataset(row_count=300)
        app.train_model.clear()
        with patch('pandas.util.hash_pandas_object', side_effect=TypeError('unsupported pandas hashing')):
            first = app.train_model(data)
            second = app.train_model(data.copy())
            self.assertIs(first, second)
            changed = data.copy()
            changed.loc[changed.index[0], 'weight_lbs'] += 1
            third = app.train_model(changed)
            self.assertIsNot(first, third)

    def test_digest_tracks_schema_and_rows_beyond_streamlit_sample(self):
        data = pd.DataFrame({'value': range(12000)})
        digest = app.training_data_digest(data)
        self.assertEqual(digest, app.training_data_digest(data.copy()))
        changed = data.copy()
        changed.loc[11999, 'value'] = -1
        self.assertNotEqual(digest, app.training_data_digest(changed))
        self.assertNotEqual(digest, app.training_data_digest(data.astype(float)))

if __name__ == '__main__': unittest.main()
