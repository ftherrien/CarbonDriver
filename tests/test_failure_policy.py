import math
from unittest.mock import patch

import pandas as pd
import pytest
import torch

from carbondriver import GDEOptimizer, OptimizerTrainingError


def optimizer(policy="random"):
    return GDEOptimizer(
        model_name="GP",
        quantity="objective",
        input_labels=["temperature", "ink"],
        output_labels=["objective"],
        bounds=torch.tensor([[40.0, 20.0], [80.0, 60.0]]),
        config={"make_plots": False, "model_failure_policy": policy},
    )


def observations():
    return pd.DataFrame(
        {
            "temperature": [40.0, 60.0],
            "ink": [20.0, 40.0],
            "objective": [10.0, 20.0],
        }
    )


def test_raise_policy_stops_without_recommending():
    driver = optimizer("raise")
    failure = RuntimeError("You must train on the training inputs")
    with patch.object(driver, "get_predictor", side_effect=failure):
        with pytest.raises(OptimizerTrainingError, match="No recommendation"):
            driver.step(observations())


def test_default_random_policy_stays_inside_bounds():
    driver = optimizer()
    failure = torch._C._LinAlgError("singular covariance")
    with patch.object(driver, "get_predictor", side_effect=failure):
        with pytest.warns(RuntimeWarning, match="random exploration"):
            score, suggestion = driver.step(observations())
    assert math.isnan(score)
    assert 40.0 <= suggestion["temperature"] <= 80.0
    assert 20.0 <= suggestion["ink"] <= 60.0


def test_random_candidate_policy_returns_a_dataframe_index():
    driver = optimizer()
    candidates = pd.DataFrame(
        {"temperature": [50.0], "ink": [30.0]},
        index=["only-candidate"],
    )
    failure = RuntimeError("cholesky_cpu failed")
    with patch.object(driver, "get_predictor", side_effect=failure):
        with pytest.warns(RuntimeWarning, match="random exploration"):
            score, selected = driver.step_within_data(observations(), candidates)
    assert math.isnan(score)
    assert selected == "only-candidate"


def test_invalid_policy_is_rejected():
    with pytest.raises(ValueError, match="model_failure_policy"):
        optimizer("unsupported")


def test_empty_candidate_table_is_rejected():
    driver = optimizer()
    candidates = pd.DataFrame(columns=["temperature", "ink"])
    with pytest.raises(ValueError, match="at least one candidate"):
        driver.step_within_data(observations(), candidates)
