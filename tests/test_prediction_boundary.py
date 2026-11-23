"""Regression tests for the prediction boundary, without launching Streamlit."""
import ast
from pathlib import Path
import unittest

import numpy as np

from generate_data import generate_dataset


def load_model_functions():
    # Execute the real model/recommendation definitions without dashboard imports
    # or Streamlit caching; pandas and scikit-learn remain real dependencies.
    source = Path(__file__).resolve().parents[1] / "app.py"
    tree = ast.parse(source.read_text())
    nodes = []
    wanted = {"train_model", "recommendation_text", "build_recommendations"}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (
            node.module == "__future__" or node.module == "typing"
            or node.module.startswith("sklearn")
        ):
            nodes.append(node)
        elif isinstance(node, ast.Import) and any(a.name == "pandas" for a in node.names):
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "FEATURE_COLUMNS" for t in node.targets
        ):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted:
            node.decorator_list = []
            nodes.append(node)
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace, tree


class PredictionBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functions, cls.tree = load_model_functions()
        cls.data = generate_dataset(row_count=600, seed=42)
        cls.artifacts = cls.functions["train_model"](
            cls.data.drop(columns=["damage_source"])
        )

    def test_recommendation_rate_is_weighted_by_handled_cases(self):
        data = self.data.iloc[[0, 0]].copy()
        data['damaged_cases'] = [1, 9]
        data['cases_handled'] = [2, 98]
        data['damage_rate'] = [.5, 9 / 98]
        result = self.functions['build_recommendations'](data)
        self.assertEqual(result.iloc[0]['damage_rate'], .1)

    def test_model_excludes_incident_outcomes(self):
        columns = set(self.artifacts["model"].feature_names_in_)
        self.assertTrue(columns.isdisjoint({
            "damage_source", "damaged_cases", "damage_rate",
            "damage_cost", "damage_type", "high_damage",
        }))
        self.assertFalse(any(
            "damage_source" in name
            for name in self.artifacts["model"].named_steps[
                "preprocessor"
            ].get_feature_names_out()
        ))

    def test_predictions_ignore_recorded_damage_source(self):
        model = self.artifacts["model"]
        rows = self.data.head(20).copy()
        expected = model.predict_proba(rows.drop(columns=["damage_source"]))
        rows["damage_source"] = "Pallet collapse"
        np.testing.assert_array_equal(expected, model.predict_proba(rows))
        rows["damage_source"] = "Driver backing up"
        np.testing.assert_array_equal(expected, model.predict_proba(rows))

    def test_prevention_guidance_needs_no_incident_source(self):
        row = self.data.iloc[0].to_dict()
        row.pop("damage_source")
        row["congestion_level"] = "High"
        text = self.functions["recommendation_text"](row)
        self.assertIn("Reduce congestion", text)
        self.assertNotIn("backing procedure audit", text)

    def test_historical_guidance_retains_recorded_source(self):
        data = self.data.head(10).copy()
        data["damage_source"] = "Driver backing up"
        recommendations = self.functions["build_recommendations"](data)
        self.assertTrue(recommendations["recommended_action"].str.contains(
            "backing procedure audit", regex=False
        ).all())

    def test_prediction_form_has_no_damage_source_input(self):
        form = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "show_prediction_tool")
        names = {n.id for n in ast.walk(form) if isinstance(n, ast.Name)}
        literals = {n.value for n in ast.walk(form)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        self.assertNotIn("damage_source", names)
        self.assertNotIn("damage_source", literals)
        self.assertNotIn("Damage Source", literals)


if __name__ == "__main__":
    unittest.main()

