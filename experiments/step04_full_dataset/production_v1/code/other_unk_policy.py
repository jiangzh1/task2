"""Production v1 terminal policy for emotion2vec Other/UNK predictions."""

FORMAL_LABELS = ('happy', 'sad', 'angry', 'surprised', 'disgusted', 'fearful', 'neutral')


def resolve_other_or_unk(scores: dict[str, float], raw_label: str, threshold: float = 0.30) -> tuple[str, bool]:
    """Return (label, retry_required) following the production-v1 policy."""
    if raw_label not in {'other', 'unk', '<unk>'}:
        return raw_label, False
    best = max(FORMAL_LABELS, key=lambda label: scores.get(label, 0.0))
    if scores.get(best, 0.0) > threshold:
        return best, False
    return 'other', True
