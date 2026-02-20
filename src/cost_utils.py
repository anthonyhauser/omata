

#cost estimates
def estimate_openai_cost(usage: dict, model: str) -> dict:
    """
    Estimate the approximate cost of an OpenAI API call.

    Parameters:
        usage : dict
            The 'usage' field from the OpenAI API response.
        model : str
            The model used (e.g., "gpt-4", "gpt-4.1-mini", "gpt-3.5-turbo")

    Returns:
        dict with prompt_tokens, completion_tokens, total_tokens, estimated_cost_usd
    """
    # Token counts
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    # Model rates per 1M tokens (USD),
    rates = { "gpt-4o": {"prompt": 2.50, "completion": 10.00},   #add cached input (but very small)       #https://platform.openai.com/docs/pricing
              "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
              "gpt-4.1": {"prompt": 2.00, "completion": 8.00},
              "gpt-5.2": {"prompt": 1.75, "completion": 14.00},
              "gpt-5": {"prompt": 1.25, "completion": 10.00},
              "gpt-5-mini": {"prompt": 0.25, "completion": 2.00},
              "gpt-5-nano": {"prompt": 0.05, "completion": 0.40}}

    model_key = model.lower()
    if model_key not in rates:
        raise ValueError(f"Unknown model key: {model_key}, {model}")

    # Compute cost
    
    input_cost = (prompt_tokens / 1_000_000) * rates[model_key]["prompt"]
    output_cost = (completion_tokens / 1_000_000) * rates[model_key]["completion"]
    
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost
    }

