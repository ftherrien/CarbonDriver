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
    # LLM-based active learning settings (used when model_name="LLM")
    "llm_api": "gemini",          # "gemini", "openai", or "claude"
    "llm_model": "gemini-3.5-flash",
    "llm_api_key": None,
    "llm_experiment_context": (
        "You are an expert scientist helping to optimize an experiment. "
        "Use your domain knowledge along with the experimental data provided to make informed suggestions."
    ),
    "llm_max_attempts": 3,
}
