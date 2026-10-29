"""Reproduce the fixed final model from official training inputs on CPU."""

import time

STARTED = time.perf_counter()

import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True,
                        help='Fixed final configuration, artifacts/final/config.json.')
    parser.add_argument('--run-dir', type=Path, required=True,
                        help='New output directory below runs/.')
    parser.add_argument('--data-dir', type=Path,
                        help='Official data directory (default code/data).')
    parser.add_argument('--reproduce-comparisons', action='store_true',
                        help='Reproduce the fixed three-model and nine-feature train-only CV comparisons (36 fits).')
    args = parser.parse_args(argv)
    # Apply the CPU thread limit before importing NumPy or SciPy.
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '4'
    options = {} if args.data_dir is None else {'data_dir': args.data_dir}
    if args.reproduce_comparisons:
        from mp2.comparisons import run_comparisons
        report = run_comparisons(args.config, args.run_dir, started=STARTED, **options)
        keys = ('status', 'informative_questions', 'fit_count', 'train_seconds', 'comparisons_file')
    else:
        from mp2.final_training import train_final
        report = train_final(args.config, args.run_dir, started=STARTED, **options)
        keys = ('status', 'training_support', 'train_seconds', 'model_bundle')
    print(json.dumps({key: report[key] for key in keys}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
