default_config = {
    "run_name": "out",
    "num_iter": 101,
    "make_plots": False,
    "normalize_inputs": True,
    "normalize_outputs": False,
    "EI_reference": "max",
    "UCB_beta": 1.0,
    "system_phase": "gas",
    "property_name": "FE (Eth)",
    "data_file": None,
    "acquisition": "EI",
    "extra_sink": False,
    "constant_J_in": False,
    "current_density": None,
    "zero_eps_thickness": None,
    "t_CO2": None,
    "torch_seed": None,  # seed torch before fitting PhModel/GP+Ph, for reproducibility
    "propose_random_on_fallback": True,
    # LLM-based active learning settings (used when model_name="LLM")
    "llm_api": "claude",  # "gemini" (not yet supported), "openai", or "claude"
    "llm_model": "claude-haiku-4-5-20251001",
    "llm_api_key": None,
    "llm_experiment_context": (
        "You are an expert scientist helping to optimize an experiment. "
        "Use your domain knowledge along with the experimental data provided to make informed suggestions."
    ),
    "llm_max_tokens": 1024,
    "similarity_tolerance": 1e-5,
    "max_attempts": 3,  # retries for LLM calls and for proposing a point already in the dataset
    "allow_repeated_experiments": False,  # if True, skip retries and return a candidate even if it duplicates an existing point
    "num_restarts": 20,
    "raw_samples": 200,
    "num_models": 50,
    "batch_fraction": 0.5,
    "dropout": 0.1,
    "variance_scaling": True,
    "per_output_scaling": False,
    "ph_frac": 0.5,
    "zero_eps_thickness_bounds": [1e-6, 1e-5],
    "lr": 1e-3,
    "tol": 1e-5,
    "early_stopping": False,
}
