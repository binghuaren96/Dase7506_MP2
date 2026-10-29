"""Reproduce one fixed final model from official training inputs only.

This entry does not load development caches, validation, test, previous runs or
Git metadata. It performs one deterministic fit and leaves the published assets
unchanged. The official manifest checks the three training files before fitting.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from common import DATA, PROTOCOL, check_data, load_inputs, parse_answer, read_jsonl
from mp2 import group_linear
from mp2.selection import build_groups


ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = ROOT / 'runs'


FINAL_CONFIG = {'method': 'group_linear_bt', 'variant': 'group_shape_bt', 'l2': 0.01}
TRAIN_FILES = ('candidates/gsm8k_train.jsonl', 'candidates/gsm8k_train.npz', 'gsm8k_train.jsonl')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_output_directory(run_dir):
    requested = Path(run_dir).expanduser()
    output, runs = requested.resolve(), RUNS_ROOT.resolve()
    if not output.is_relative_to(runs) or output == runs:
        raise ValueError("Use a new output directory strictly inside this project's runs/.")
    if requested.is_symlink() or output.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output}')
    return output


def validate_config(config):
    """Keep the submitted method fixed, rather than exposing a new search."""
    if (not isinstance(config, dict) or set(config) != set(FINAL_CONFIG)
            or any(type(config[key]) is not type(expected) or config[key] != expected
                   for key, expected in FINAL_CONFIG.items())):
        raise ValueError('Expected exactly the fixed final configuration: group_linear_bt, group_shape_bt, l2=0.01.')


def training_labels(bank, references):
    """Bind multi-positive labels to bank order and find informative questions."""
    if not bank or len(bank) != len(references):
        raise ValueError('Training bank and references must be nonempty and have identical lengths.')
    labels = np.zeros((len(bank), 4), dtype=np.bool_)
    for index, (row, reference) in enumerate(zip(bank, references)):
        if row['id'] != reference['id'] or row['question'] != reference['question']:
            raise ValueError(f'Training reference/bank alignment mismatch at row {index}.')
        answer = parse_answer(reference['answer'])
        if answer is None:
            raise ValueError(f'Unparseable training reference at row {index}.')
        if len(row['candidates']) != 4:
            raise ValueError('Every training question must have four candidates.')
        labels[index] = [value is not None and value == answer
                         for value in (parse_answer(candidate['text']) for candidate in row['candidates'])]
        if not labels[index].any():
            raise ValueError(f'No correct training candidate at row {index}.')
    hard = build_groups(bank).any(axis=2).sum(axis=1) >= 2
    return labels, hard


def _write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write('\n')


def train_final(config_path, run_dir, *, data_dir=DATA, started=None):
    """Fit once on train disagreement groups; save a new bundle and one summary."""
    tick = time.perf_counter() if started is None else started
    configuration = json.loads(Path(config_path).read_text(encoding='utf-8'))
    validate_config(configuration)
    output = validate_output_directory(run_dir)
    data = Path(data_dir).resolve()
    check_data(data, ('train',), inputs_only=False)
    bank, arrays = load_inputs('train', data)
    references = read_jsonl(data / 'gsm8k_train.jsonl')
    labels, hard = training_labels(bank, references)
    indices = np.flatnonzero(hard)
    if not len(indices):
        raise ValueError('At least one valid answer-disagreement training question is required.')
    input_hashes = {name: sha256_file(data / name) for name in TRAIN_FILES}
    # Create a new terminal directory atomically; existing outputs are refused.
    output.mkdir(parents=True, exist_ok=False)
    fit_started = time.perf_counter()
    bundle, diagnostics = group_linear.fit(
        [bank[int(index)] for index in indices], arrays['embeddings'][indices], labels[indices],
        configuration['variant'], l2=configuration['l2'],
    )
    train_seconds = time.perf_counter() - fit_started
    checkpoint = output / 'model_bundle'
    checkpoint.mkdir()
    _write_json(checkpoint / 'config.json', configuration)
    np.savez(checkpoint / 'weights.npz', **bundle)
    import scipy

    report = {
        'status': 'complete',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': PROTOCOL,
        'config': configuration,
        'training_support': {
            'complete': len(bank),
            'all_four_correct': int(labels.all(axis=1).sum()),
            'not_all_four_correct': int((~labels.all(axis=1)).sum()),
            'disagreement': len(indices),
            'non_all_correct_single_valid_group': int((~labels.all(axis=1) & ~hard).sum()),
        },
        'diagnostics': diagnostics,
        'train_seconds': train_seconds,
        'elapsed_seconds': time.perf_counter() - tick,
        'software': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
        'random_seed': None,
        'initialization': 'zero coefficients; deterministic L-BFGS-B',
        'reference_access': 'train only; no validation, test, audit or cached labels',
        'input_sha256': input_hashes,
        'model_bundle': str(checkpoint),
        'assets': {name: {'sha256': sha256_file(checkpoint / name),
                          'bytes': (checkpoint / name).stat().st_size}
                   for name in ('config.json', 'weights.npz')},
    }
    _write_json(output / 'training.json', report)
    return report
