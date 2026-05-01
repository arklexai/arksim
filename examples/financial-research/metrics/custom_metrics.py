def score_has_both_sides(agent_output: dict) -> float:
    bull = agent_output.get("bull_case", [])
    bear = agent_output.get("bear_case", [])
    if bull and bear:
        return 1.0
    if bull or bear:
        return 0.5
    return 0.0


def score_has_risks(agent_output: dict) -> float:
    risks = agent_output.get("key_risks", [])
    return 1.0 if risks else 0.0


def score_has_sources(agent_output: dict) -> float:
    sources = agent_output.get("sources", [])
    return 1.0 if len(sources) >= 2 else 0.0


def evaluate(agent_output: dict, scenario: dict) -> dict:
    return {
        "has_both_sides": score_has_both_sides(agent_output),
        "has_risks": score_has_risks(agent_output),
        "has_sources": score_has_sources(agent_output),
    }