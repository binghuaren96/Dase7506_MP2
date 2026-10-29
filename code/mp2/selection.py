"""Numeric answer grouping using the unchanged official parser."""

from common import parse_answer


def build_groups(bank):
    """Build bool [N, 4 groups, 4 candidates] in answer first-occurrence order.

    Invalid answers have no membership. Empty banks and rows without any
    valid answer are legal for inference; those rows yield no training pairs.
    """
    import numpy as np

    if not isinstance(bank, (list, tuple)):
        raise ValueError('bank must be a sequence of question dictionaries')
    membership = np.zeros((len(bank), 4, 4), dtype=bool)
    for row_index, row in enumerate(bank):
        if not isinstance(row, dict):
            raise ValueError('each bank row must be a question dictionary')
        candidates = row.get('candidates')
        if not isinstance(candidates, (list, tuple)) or len(candidates) != 4:
            raise ValueError('each question must contain exactly four candidates')
        answer_groups = {}
        for position, candidate in enumerate(candidates):
            if not isinstance(candidate, dict) or not isinstance(candidate.get('text'), str):
                raise ValueError('each candidate must contain a text string')
            answer = parse_answer(candidate['text'])
            if answer is not None:
                group = answer_groups.setdefault(answer, len(answer_groups))
                membership[row_index, group, position] = True
    return membership
