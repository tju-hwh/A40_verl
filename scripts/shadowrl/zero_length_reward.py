def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kwargs):
    """Cheap reward used for throughput benchmarking.

    It still runs through VERL's reward manager, but avoids dataset-specific
    correctness checks so the three prompt datasets are comparable.
    """
    text = solution_str or ""
    return {
        "score": min(len(text) / 512.0, 1.0),
        "response_chars": len(text),
    }
