default_config = {
    "run_name": "out",
    "num_iter": 101,
    "make_plots": False,
    "normalize_inputs": True,
    "normalize_outputs": False,
    "EI_reference": "max",
    "UCB_beta": 1.0,
    "system_phase": None,
    "dataset": "gas",
    "property_name": "FE (Eth)",
    "data_file": None,
    "acquisition": "EI",
    "extra_sink": False,
    "constant_J_in": False,
    "current_density": None,
    "zero_eps_thickness": None,
    "t_CO2": None,
    "torch_seed": None,  # seed torch before fitting PhModel/GP+Ph, for reproducibility
    "propose_random_when_underdetermined": True,
    # LLM-based active learning settings (used when model_name="LLM")
    "llm_api": "claude",          # "gemini" (not yet supported), "openai", or "claude"
    "llm_model": "claude-haiku-4-5-20251001",
    "llm_api_key": None,
    "llm_experiment_context": (
        "You are an expert scientist helping to optimize an experiment. "
        "Use your domain knowledge along with the experimental data provided to make informed suggestions."
    ),
    "llm_max_tokens": 1024,
    "llm_max_attempts": 3,
    "similarity_tolerance": 1e-5,
}
