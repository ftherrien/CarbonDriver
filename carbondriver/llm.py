import json
from typing import Optional

import torch


def create_prompt(
    opt,
    mode: str,
    bounds: Optional[torch.Tensor] = None,
    possible_data=None,
) -> str:
    """Build the user prompt from history, data, and task description."""
    direction = "maximiz" if opt.maximize else "minimiz"

    history_str = ""
    if opt.llm_history:
        entries = "\n".join(
            f"  Step {e['step']}: suggested {e['suggestion']} — reason: {e['reason']}"
            for e in opt.llm_history
        )
        history_str = f"Your previous suggestions and reasoning:\n{entries}\n\n"

    data_str = f"Experimental data collected so far:\n{opt.df.to_string()}\n\n"

    if mode == "step":
        raw_bounds = opt.bounds if bounds is None else bounds
        bounds_str = "\n".join(
            f"  {label}: [{raw_bounds[0, i].item():.4g}, {raw_bounds[1, i].item():.4g}]"
            for i, label in enumerate(opt.input_labels)
        )
        return (
            f"{history_str}"
            f"{data_str}"
            f"Input parameters and their allowed ranges:\n{bounds_str}\n\n"
            f"Suggest the next experiment that will lead to {direction}ing '{opt.quantity}' "
            f"in the fewest number of experiments.\n"
            f"Respond with ONLY a JSON object with:\n"
            f"  parameter names mapped to their suggested values\n"
            f'  "reason": brief explanation\n'
            f'Example: {json.dumps({**{l: 0.0 for l in opt.input_labels}, "reason": "..."})}'
        )
    else:  # step_within_data
        return (
            f"{history_str}"
            f"{data_str}"
            f"Candidate experiments to choose from (index on the left):\n{possible_data.to_string()}\n\n"
            f"Select the candidate that will lead to {direction}ing '{opt.quantity}' "
            f"in the fewest number of experiments.\n"
            f"Respond with ONLY a JSON object with:\n"
            f'  "index": index of the best candidate in the dataframe (leftmost column)\n'
            f'  "reason": brief explanation\n'
            f'Example: {{"index": 0, "reason": "..."}}'
        )


def read_response(text: str) -> dict:
    """Strip markdown code fences from LLM output and parse as JSON."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def llm_suggest(
    opt,
    mode: str,
    bounds: Optional[torch.Tensor] = None,
    possible_data=None,
    error_message: Optional[str] = None,
) -> dict:
    """
    Call the configured LLM API to suggest the next experiment.

    :param opt: GDEOptimizer instance
    :param mode: "step" for free-form suggestion within bounds, or "step_within_data" to select from candidates
    :param bounds: tensor of shape (2, d) with parameter bounds (used in "step" mode)
    :param possible_data: DataFrame of candidate experiments (used in "step_within_data" mode)
    :returns: parsed JSON dict from the LLM response
    """
    api = opt.config.get("llm_api", "gemini")
    system = opt.config["llm_experiment_context"]
    if error_message is not None:
        user = f"Parsing of your previous output failed with error: {error_message}\n\n"
    else:
        user = create_prompt(opt, mode, bounds=bounds, possible_data=possible_data)

    api_key = opt.config.get("llm_api_key", None)

    if api == "gemini":
        raise NotImplementedError(
            "The 'gemini' llm_api is not currently supported (rate-limited/unavailable); use 'openai' or 'claude' instead."
        )

    elif api == "openai":
        from openai import OpenAI

        response = OpenAI(api_key=api_key).chat.completions.create(
            model=opt.config["llm_model"],
            messages=[{"role": "system", "content": system}]
            + opt.raw_messages
            + [{"role": "user", "content": user}],
        )
        text = response.choices[0].message.content

    elif api == "claude":
        from anthropic import Anthropic

        response = Anthropic(api_key=api_key).messages.create(
            model=opt.config["llm_model"],
            max_tokens=opt.config.get("llm_max_tokens", 1024),
            system=system,
            messages=opt.raw_messages + [{"role": "user", "content": user}],
        )
        # Claude may return thinking blocks before the final text block.
        # Do not assume response.content[0] is a TextBlock.
        text_blocks = [
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        ]
        if not text_blocks:
            raise ValueError("Claude response contained no text block")
        text = "\n".join(text_blocks)

    else:
        raise ValueError(
            f"Unsupported llm_api '{api}'. Choose 'gemini', 'openai', or 'claude'."
        )

    opt.raw_messages.extend(
        [{"role": "user", "content": user}, {"role": "assistant", "content": text}]
    )

    return read_response(text)
