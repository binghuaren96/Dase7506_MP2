"""Frozen answer-group Bradley--Terry selection; no references or network access."""

import json
from pathlib import Path

import numpy as np

from mp2 import group_linear


def predict(bank, arrays, checkpoint):
    """Return one candidate index per supplied question, in the supplied order."""
    if checkpoint is None or not Path(checkpoint).is_dir():
        raise ValueError('A model bundle directory containing config.json and weights.npz is required.')
    directory = Path(checkpoint)
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    if (not isinstance(config, dict) or config.get('method') != 'group_linear_bt'
            or config.get('variant') not in group_linear.VARIANTS):
        raise ValueError('Unsupported model configuration.')
    with np.load(directory / 'weights.npz', allow_pickle=False) as archive:
        bundle = {key: archive[key] for key in archive.files}
    variant = np.asarray(bundle.get('variant'))
    if variant.shape != () or variant.dtype.kind not in 'US' or str(variant.item()) != config['variant']:
        raise ValueError('Configuration and checkpoint variants differ.')
    return group_linear.predict(bank, arrays['embeddings'], bundle)
