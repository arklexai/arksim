from typing import Any


# flip roles in convo history, only keep role and content
def flip_hist_content_only(hist: list[dict[str, Any]]) -> list[dict[str, Any]]:
    new_hist = []
    for turn in hist:
        if turn["role"] == "system":
            continue
        elif turn["role"] == "user":
            new_hist.append({"role": "assistant", "content": turn["content"]})
        else:
            new_hist.append({"role": "user", "content": turn["content"]})
    return new_hist


# flip roles in convo history, keep all other keys the same
def flip_hist(hist: list[dict[str, Any]]) -> list[dict[str, Any]]:
    new_hist = []
    for turn in hist:
        if "role" not in turn:
            new_hist.append(turn)
        elif turn["role"] == "system":
            continue
        elif turn["role"] == "user":
            new_hist.append({**turn, "role": "assistant"})
        else:
            new_hist.append({**turn, "role": "user"})
    return new_hist
