"""Fixed train-only three-fold comparisons for the report: 36 fits, no search."""

import json
from pathlib import Path
import time

import numpy as np

from common import DATA, PROTOCOL, check_data, load_inputs, read_jsonl
from mp2 import final_training, group_linear


FOLD_SEED = 20261028
FOLD_COUNT = 3
MAIN_VARIANTS = ('candidate_bt', 'group_bt', 'group_shape_bt')


def make_folds(bank, labels, hard):
    """Stratify all train questions before filtering the informative support."""
    labels, hard = np.asarray(labels), np.asarray(hard)
    if (labels.dtype != np.bool_ or labels.shape != (len(bank), 4)
            or hard.dtype != np.bool_ or hard.shape != (len(bank),)
            or len(bank) < FOLD_COUNT or not labels.any(axis=1).all()):
        raise ValueError('Aligned multi-positive labels and disagreement flags are required.')
    rng = np.random.default_rng(FOLD_SEED)
    assignment = np.empty(len(bank), dtype=np.int64)
    offset = 0
    for informative in (False, True):
        for correct_count in range(1, 5):
            indices = np.flatnonzero((hard == informative) & (labels.sum(axis=1) == correct_count))
            rng.shuffle(indices)
            assignment[indices] = (np.arange(len(indices)) + offset) % FOLD_COUNT
            offset = (offset + len(indices)) % FOLD_COUNT
    first = {}
    for index, row in enumerate(bank):
        if row['question'] in first:
            assignment[index] = assignment[first[row['question']]]
        else:
            first[row['question']] = index
    return assignment


def _metrics(choices, labels, folds, anchor=None):
    correct = labels[np.arange(len(labels)), choices]
    result = dict(correct=int(correct.sum()), n=len(labels), accuracy=100 * float(correct.mean()),
                  fold_correct=[int(correct[folds == fold].sum()) for fold in range(FOLD_COUNT)])
    if anchor is not None:
        baseline = labels[np.arange(len(labels)), anchor]
        delta = correct.astype(np.int8) - baseline.astype(np.int8)
        result['paired_vs_final'] = dict(gains=int((correct & ~baseline).sum()),
            losses=int((~correct & baseline).sum()), net=int(delta.sum()),
            fold_nets=[int(delta[folds == fold].sum()) for fold in range(FOLD_COUNT)])
    return result


def run_comparisons(config_path, run_dir, *, data_dir=DATA, started=None):
    """Rebuild folds, fit three representations and nine drop-one ablations."""
    tick = time.perf_counter() if started is None else started
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    final_training.validate_config(config)
    output = final_training.validate_output_directory(run_dir)
    data = Path(data_dir).resolve()
    check_data(data, ('train',), inputs_only=False)
    bank, arrays = load_inputs('train', data)
    labels, hard = final_training.training_labels(bank, read_jsonl(data / 'gsm8k_train.jsonl'))
    folds = make_folds(bank, labels, hard)
    informative = np.flatnonzero(hard)
    if any(not np.any(hard & (folds == fold)) or not np.any(hard & (folds != fold))
           for fold in range(FOLD_COUNT)):
        raise ValueError('Every fold requires informative training and held-out questions.')
    output.mkdir(parents=True, exist_ok=False)
    descriptors = [(variant, variant, ()) for variant in MAIN_VARIANTS]
    descriptors += [('drop_' + feature, 'group_shape_bt', (feature,))
                    for feature in group_linear.SHAPE_FEATURES]
    choices = {name: np.full(len(informative), -1, dtype=np.int64)
               for name, _, _ in descriptors}
    fit_count, train_seconds = 0, 0.0
    for fold in range(FOLD_COUNT):
        train = np.flatnonzero(hard & (folds != fold))
        heldout = np.flatnonzero(hard & (folds == fold))
        target = np.flatnonzero(folds[informative] == fold)
        training_bank = [bank[int(index)] for index in train]
        heldout_bank = [bank[int(index)] for index in heldout]
        for name, variant, dropped in descriptors:
            fit_started = time.perf_counter()
            bundle, _ = group_linear.fit(training_bank, arrays['embeddings'][train], labels[train],
                                        variant, l2=config['l2'], drop_shape_features=dropped)
            train_seconds += time.perf_counter() - fit_started
            fit_count += 1
            choices[name][target] = group_linear.predict(heldout_bank, arrays['embeddings'][heldout], bundle)
    if fit_count != 36 or any(np.any(value < 0) for value in choices.values()):
        raise RuntimeError('Fixed comparison matrix did not complete.')
    y, heldout_folds = labels[informative], folds[informative]
    anchor = choices['group_shape_bt']
    summary = {name: _metrics(value, y, heldout_folds, None if name == 'group_shape_bt' else anchor)
               for name, value in choices.items()}
    report = dict(status='complete', protocol=PROTOCOL, config=config,
        train_questions=len(bank), informative_questions=len(informative),
        fold_count=FOLD_COUNT, fold_seed=FOLD_SEED,
        folds_generated_before_informative_filter=True,
        exact_duplicate_questions_share_first_fold=True,
        fold_informative_counts=[int(np.sum(hard & (folds == fold))) for fold in range(FOLD_COUNT)],
        fit_count=fit_count, train_seconds=train_seconds, elapsed_seconds=time.perf_counter() - tick,
        input_sha256={name: final_training.sha256_file(data / name) for name in final_training.TRAIN_FILES},
        comparisons=summary, comparisons_file=str(output / 'comparisons.json'),
        reference_access='train only; no validation, test, previous runs or checkpoint access',
        note='Fixed comparisons, not parameter selection; no deployment assets are changed.')
    final_training._write_json(output / 'comparisons.json', report)
    return report
