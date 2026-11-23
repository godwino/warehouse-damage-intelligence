import unittest
from io import StringIO
import pandas as pd
from picking_review import review, sample_data, csv_bytes, read_order_csv

class PickingReviewTests(unittest.TestCase):
    def test_demo_improves_without_crossing_zones_or_changing_workload(self):
        data = sample_data()
        result, before, after = review(data)
        self.assertEqual(len(before), 4)
        self.assertEqual(len(after), 0)
        self.assertEqual(len(result), len(data))
        for _, group in result.groupby(['assignment_id', 'zone']):
            self.assertEqual(sorted(group.pick_position_current), sorted(group.pick_position_proposed))
        self.assertEqual(result.quantity.sum(), data.quantity.sum())

    def test_nonadjacent_pair_is_checked(self):
        data = sample_data().iloc[:3].copy()
        data.loc[data.index[1], 'weight_kg'] = 2
        _, before, _ = review(data)
        self.assertIn('RICE', before.later_sku.tolist())

    def test_different_assignments_and_zones_never_form_pairs(self):
        data = sample_data().iloc[:2].copy()
        data.loc[1, 'zone'] = 'other'
        self.assertTrue(review(data)[1].empty)
        data.loc[1, 'zone'] = data.loc[0, 'zone']
        data.loc[1, 'assignment_id'] = 'other'
        self.assertTrue(review(data)[1].empty)

    def test_bad_inputs_rejected(self):
        for column, value in [('quantity', 0), ('quantity', 1.5), ('weight_kg', float('inf')),
                              ('crushability', 1.1), ('length_cm', -1), ('sku', None), ('pick_position', 'oops')]:
            with self.subTest(column=column, value=value):
                data = sample_data().copy()
                data[column] = data[column].astype(object)
                data.loc[0, column] = value
                with self.assertRaises(ValueError): review(data)
        for column in ['sku', 'pick_position']:
            data = sample_data()
            data.loc[1, column] = data.loc[0, column]
            with self.assertRaises(ValueError): review(data)
        with self.assertRaises(ValueError): review(pd.DataFrame())

    def test_already_robust_sequence_stays_unchanged(self):
        data = sample_data().iloc[:2].copy()
        data['pick_position'] = [2, 1]
        result, before, after = review(data)
        self.assertTrue(before.empty and after.empty)
        self.assertTrue((result.pick_position_current == result.pick_position_proposed).all())

    def test_csv_preserves_identifiers_and_neutralizes_formulas(self):
        content = csv_bytes(pd.DataFrame({'sku': ['=1+1', '001'], 'quantity': [2, 3]})).decode('utf-8-sig')
        self.assertIn("'=1+1", content)
        self.assertIn('001', content)

    def test_csv_keeps_literal_identifiers_and_rejects_oversize(self):
        data = sample_data().iloc[:1].copy()
        data['sku'] = '001'
        data['zone'] = 'NA'
        parsed = read_order_csv(StringIO(data.to_csv(index=False)))
        self.assertEqual(parsed.loc[0, 'sku'], '001')
        self.assertEqual(parsed.loc[0, 'zone'], 'NA')
        review(parsed)
        large = pd.concat([data] * 2002, ignore_index=True)
        parsed = read_order_csv(StringIO(large.to_csv(index=False)))
        self.assertEqual(len(parsed), 2001)
        with self.assertRaises(ValueError): review(parsed)

if __name__ == '__main__': unittest.main()
