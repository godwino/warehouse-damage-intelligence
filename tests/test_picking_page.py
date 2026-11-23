from pathlib import Path
import unittest
from streamlit.testing.v1 import AppTest

class PickingPageTests(unittest.TestCase):
    def test_demo_and_assignment_selector(self):
        page = Path(__file__).resolve().parents[1] / 'pages/1_Picking_Order_Review.py'
        app = AppTest.from_file(str(page)).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual([m.value for m in app.metric[:2]], ['4', '0'])
        self.assertEqual(len(app.dataframe), 4)
        app.selectbox[0].select(('A002', 'Zone 2')).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(set(app.dataframe[1].value['sku']), {'CEREAL', 'JUICE'})
        self.assertEqual(len(app.get('download_button')), 4)
        app.checkbox[0].uncheck().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 0)

if __name__ == '__main__': unittest.main()
