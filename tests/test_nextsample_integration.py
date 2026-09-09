import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from carbondriver import (
    GDEOptimizer,
    OptimizerTrainingError,
    PhysicsOutputAdapter,
    _candidate_scores,
)
from carbondriver.domains import PhysicsDomain
from carbondriver.loaders import _validate_electrode_area


class FixedPhysicsModel(torch.nn.Module):
    def forward(self, x):
        return torch.tensor([[0.2, 0.6]], dtype=x.dtype).expand(x.shape[0], -1)


class CandidateScoreTests(unittest.TestCase):
    def test_supported_score_shapes(self):
        expected = torch.tensor([0.2, 0.8, 0.1])
        cases = [
            (expected, 0),
            (expected[:, None], 0),
            (torch.stack([expected * 0, expected], -1), 1),
        ]
        for values, index in cases:
            with self.subTest(shape=values.shape):
                torch.testing.assert_close(_candidate_scores(values, index, 3), expected)

    def test_invalid_score_shapes_are_rejected(self):
        cases = [
            (torch.ones(2, 3), 0, 3),
            (torch.ones(3, 1, 1), 0, 3),
            (torch.ones(3, 2), 2, 3),
            (torch.ones(3), -1, 3),
            (torch.tensor(1.0), 0, 3),
        ]
        for values, index, count in cases:
            with self.subTest(shape=values.shape), self.assertRaises(RuntimeError):
                _candidate_scores(values, index, count)

    def test_step_within_selects_highest_scoring_dataframe_row(self):
        optimizer = GDEOptimizer(
            model_name="GP",
            quantity="objective",
            input_labels=["temperature", "ink"],
            output_labels=["objective"],
            config={"make_plots": False},
        )
        training = pd.DataFrame(
            {
                "temperature": [40.0],
                "ink": [20.0],
                "objective": [10.0],
            }
        )
        candidates = pd.DataFrame(
            {
                "temperature": [50.0, 60.0, 70.0],
                "ink": [30.0, 40.0, 50.0],
            },
            index=["condition-a", "condition-b", "condition-c"],
        )
        stats = pd.DataFrame({"loss": [0.0], "nll": [0.0]})

        with patch.object(
            optimizer, "get_predictor", return_value=(object(), stats)
        ), patch.object(
            optimizer,
            "_get_data_tensors",
            return_value=(torch.zeros(3, 2), torch.empty(3, 0)),
        ), patch.object(
            optimizer,
            "_get_acquisition_function",
            return_value=lambda _: torch.tensor([0.1, 0.9, 0.2]),
        ):
            score, selected = optimizer.step_within_data(training, candidates)

        self.assertEqual(selected, "condition-b")
        self.assertAlmostEqual(score, 0.9, places=6)


class FailureSafetyTests(unittest.TestCase):
    @staticmethod
    def optimizer():
        return GDEOptimizer(
            model_name="GP",
            quantity="objective",
            input_labels=["temperature", "ink"],
            output_labels=["objective"],
            bounds=torch.tensor([[40.0, 20.0], [80.0, 60.0]]),
            config={"make_plots": False},
        )

    @staticmethod
    def training_data():
        return pd.DataFrame(
            {
                "temperature": [40.0, 60.0],
                "ink": [20.0, 40.0],
                "objective": [10.0, 20.0],
            }
        )

    def test_step_never_replaces_fit_failure_with_random_suggestion(self):
        optimizer = self.optimizer()
        with patch.object(
            optimizer,
            "get_predictor",
            side_effect=RuntimeError("You must train on the training inputs"),
        ):
            with self.assertRaisesRegex(OptimizerTrainingError, "No recommendation"):
                optimizer.step(self.training_data())

    def test_step_within_never_replaces_fit_failure_with_random_index(self):
        optimizer = self.optimizer()
        candidates = pd.DataFrame(
            {"temperature": [50.0], "ink": [30.0]},
            index=["only-candidate"],
        )
        with patch.object(
            optimizer,
            "get_predictor",
            side_effect=RuntimeError("cholesky_cpu failed"),
        ):
            with self.assertRaisesRegex(OptimizerTrainingError, "No candidate"):
                optimizer.step_within_data(self.training_data(), candidates)

    def test_empty_candidate_table_is_rejected(self):
        optimizer = self.optimizer()
        candidates = pd.DataFrame(columns=["temperature", "ink"])
        with self.assertRaisesRegex(ValueError, "at least one candidate"):
            optimizer.step_within_data(self.training_data(), candidates)


class PhysicsContractTests(unittest.TestCase):
    labels = [
        "temperature",
        "Ink mass",
        "Ag weight",
        "zero_eps_thickness",
        "current_density",
    ]
    bounds = torch.tensor(
        [[30.0, 24.0, 2.0, 0.0, 99.0], [150.0, 96.0, 12.0, 1.0, 101.0]]
    )
    config = {
        "current_density": 100.0,
        "electrode_area_cm2": 4.0,
        "physics_mass_source": "Ag weight",
    }

    def test_fixed_and_derived_features_are_not_optimized(self):
        domain = PhysicsDomain(
            self.labels, self.bounds, self.config, {"Ag weight": 8.0}
        )
        self.assertEqual(domain.free_labels, ["temperature", "Ink mass"])
        free = torch.tensor([[[50.0, 30.0]], [[80.0, 60.0]]], requires_grad=True)
        full = domain.expand(free)
        expected_thickness = 8e-6 / (10490 * 4e-4)
        torch.testing.assert_close(full[..., 2], torch.full((2, 1), 8.0))
        torch.testing.assert_close(
            full[..., 3], torch.full((2, 1), expected_thickness)
        )
        torch.testing.assert_close(full[..., 4], torch.full((2, 1), 100.0))
        full.sum().backward()
        torch.testing.assert_close(free.grad, torch.ones_like(free))

    def test_output_adapter_selects_and_standardizes_requested_physics_output(self):
        adapter = PhysicsOutputAdapter(
            FixedPhysicsModel(),
            output_indices=[1],
            output_means=np.array([0.5]),
            output_stds=np.array([0.1]),
        )
        output = adapter(torch.zeros(1, 2))
        torch.testing.assert_close(output, torch.tensor([[1.0]]))

    def test_physics_seed_is_reapplied_for_each_fit(self):
        optimizer = GDEOptimizer(
            model_name="Ph",
            quantity="FE CO",
            input_labels=["feature"],
            output_labels=["FE CO"],
            config={
                "make_plots": False,
                "normalize_inputs": True,
                "normalize_outputs": False,
                "torch_seed": 17,
            },
        )
        optimizer.df = pd.DataFrame(
            {"feature": [0.0, 1.0], "FE CO": [0.2, 0.8]}
        )
        random_draws = []

        def fake_train(*args, **kwargs):
            random_draws.append(torch.rand(5))
            return pd.DataFrame(), object()

        with patch("carbondriver.train_model_ens", side_effect=fake_train):
            optimizer.get_predictor()
            torch.manual_seed(999)
            optimizer.get_predictor()

        torch.testing.assert_close(random_draws[0], random_draws[1])


class DataContractTests(unittest.TestCase):
    def test_negative_stride_dataframe_converts_to_tensors(self):
        optimizer = GDEOptimizer(
            model_name="GP",
            input_labels=["x"],
            output_labels=["y"],
            quantity="y",
            config={"normalize_inputs": False, "normalize_outputs": False},
        )
        raw = np.arange(12.0, dtype=float).reshape(6, 2)[::-1]
        x, y = optimizer._get_data_tensors(pd.DataFrame(raw, columns=["x", "y"]))
        np.testing.assert_array_equal(x.numpy().flatten(), raw[:, 0])
        np.testing.assert_array_equal(y.numpy().flatten(), raw[:, 1])

    def test_electrode_area_must_be_positive_and_finite(self):
        self.assertEqual(_validate_electrode_area(4.0), 4.0)
        for invalid in (0, -1, np.nan, np.inf):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _validate_electrode_area(invalid)

    def test_llm_prompt_requests_predicted_objectives(self):
        optimizer = GDEOptimizer(
            model_name="LLM",
            input_labels=["temperature"],
            output_labels=["yield"],
            quantity="yield",
            bounds=torch.tensor([[20.0], [80.0]]),
        )
        prompt = optimizer._create_prompt("step")
        self.assertIn("predicted_objectives", prompt)
        self.assertIn('"yield": 0.0', prompt)


if __name__ == "__main__":
    unittest.main()
