from unittest.mock import patch

import pandas as pd
import pytest
import torch

from carbondriver import GDEOptimizer, _candidate_scores


def test_one_dimensional_scores_preserve_candidate_axis():
    scores = torch.tensor([0.1, 0.9, 0.2])
    torch.testing.assert_close(_candidate_scores(scores, 0, 3), scores)


def test_invalid_candidate_axis_is_rejected():
    with pytest.raises(RuntimeError, match="N=3"):
        _candidate_scores(torch.ones(2, 3), 0, 3)


def test_step_within_selects_the_highest_scoring_dataframe_row():
    driver = GDEOptimizer(
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
        driver, "get_predictor", return_value=(object(), stats)
    ), patch.object(
        driver,
        "_get_data_tensors",
        return_value=(torch.zeros(3, 2), torch.empty(3, 0)),
    ), patch.object(
        driver,
        "_get_acquisition_function",
        return_value=lambda _: torch.tensor([0.1, 0.9, 0.2]),
    ):
        score, selected = driver.step_within_data(training, candidates)

    assert selected == "condition-b"
    assert score == pytest.approx(0.9)
