from pathlib import Path
import unittest
from streamlit.testing.v1 import AppTest
from generate_data import generate_dataset

class DashboardTests(unittest.TestCase):
    def test_dashboard_renders_and_predicts(self):
        root = Path(__file__).resolve().parents[1]
        data_path = root / 'data/warehouse_damage_data.csv'
        created = not data_path.exists()
        if created:
            data_path.parent.mkdir(exist_ok=True)
            generate_dataset(row_count=600).to_csv(data_path, index=False)
        try:
            app = AppTest.from_file(str(root / 'app.py')).run(timeout=60)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.get('page_link')), 1)
            self.assertTrue(any('Potential Savings' in m.label for m in app.metric))
            self.assertFalse(any(s.label == 'Damage Source' for s in app.selectbox))
            self.assertTrue(any(n.label == 'Planned Cases to Handle' for n in app.number_input))
            app.button[0].click().run(timeout=60)
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any('Prediction:' in s.value for s in app.success))
        finally:
            if created:
                data_path.unlink()

if __name__ == '__main__': unittest.main()
