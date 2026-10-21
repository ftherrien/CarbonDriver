import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--check-values",
        action="store_true",
        default=False,
        help="Assert step()/step_within_data() outputs against reference values in "
        "test_api_values.yaml (machine-specific; off by default so tests don't fail "
        "on other setups).",
    )
    parser.addoption(
        "--update-values",
        action="store_true",
        default=False,
        help="Recompute step()/step_within_data() outputs and overwrite "
        "test_api_values.yaml. Use on a new machine, or after an intentional "
        "behavior change.",
    )


@pytest.fixture
def check_values(request):
    return request.config.getoption("--check-values")


@pytest.fixture
def update_values(request):
    return request.config.getoption("--update-values")
