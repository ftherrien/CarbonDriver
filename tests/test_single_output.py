import gpytorch
import pandas as pd
import torch
from botorch.models.gp_regression import SingleTaskGP

from carbondriver import GDEOptimizer, PhysicsOutputAdapter
from carbondriver.models import MultitaskGPModel


class FixedPhysicsModel(torch.nn.Module):
    def forward(self, x):
        return torch.tensor([[0.2, 0.6]], dtype=x.dtype).expand(x.shape[0], -1)


def test_physics_output_adapter_selects_one_output():
    adapter = PhysicsOutputAdapter(FixedPhysicsModel(), output_indices=[1])
    torch.testing.assert_close(
        adapter(torch.zeros(1, 2)),
        torch.tensor([[0.6]]),
    )


def test_bicarb_output_label_alias_maps_to_physics_output():
    driver = GDEOptimizer(
        model_name="Ph",
        quantity="FE CO",
        output_labels=["FE CO"],
        config={"dataset": "bicarb"},
    )
    assert driver._physical_output_indices() == [0]


def test_single_output_gp_uses_single_task_model():
    driver = GDEOptimizer(
        model_name="GP",
        quantity="yield",
        input_labels=["temperature"],
        output_labels=["yield"],
        config={"make_plots": False},
    )
    driver.df = pd.DataFrame(
        {
            "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "yield": [0.0, 0.5, 0.8, 0.6, 0.2, 0.1],
        }
    )
    model, stats = driver.get_predictor()
    assert isinstance(model, SingleTaskGP)
    assert list(stats.columns) == ["loss", "val_loss", "nll"]


def test_multitask_gp_uses_training_output_count():
    train_x = torch.zeros(4, 2)
    train_y = torch.zeros(4, 3)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)
    model = MultitaskGPModel(train_x, train_y, likelihood)
    assert model.num_outputs == 3
    assert model.mean_module.num_tasks == 3
