"""Fold recreation and the fixed comparison API use train inputs only."""

from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mp2 import comparisons
from mp2 import final_training as final
import train


class ComparisonTests(unittest.TestCase):
    def test_folds_use_all_strata_before_filtering_and_keep_duplicate_questions_together(self):
        counts = [1, 1, 2, 2, 3, 3, 4, 4, 1, 2, 3, 1]
        labels = np.arange(4)[None, :] < np.asarray(counts)[:, None]
        hard = np.array([False] * 8 + [True] * 4)
        bank = [{'question': f'question {i}'} for i in range(12)]
        folds = comparisons.make_folds(bank, labels, hard)
        # Literal assignment for the fixed RNG/strata: includes easy-row offsets.
        np.testing.assert_array_equal(folds, [0, 1, 2, 0, 2, 1, 1, 0, 2, 1, 2, 0])
        hard_only = comparisons.make_folds(bank[8:], labels[8:], hard[8:])
        self.assertFalse(np.array_equal(folds[8:], hard_only))
        bank[10]['question'] = bank[8]['question']
        duplicate_folds = comparisons.make_folds(bank, labels, hard)
        self.assertEqual(duplicate_folds[10], duplicate_folds[8])
        np.testing.assert_array_equal(duplicate_folds[np.arange(12) != 10], folds[np.arange(12) != 10])

    def test_fixed_matrix_is_36_disjoint_train_only_fits_and_cli_routes_it(self):
        bank = []
        references = []
        for i in range(9):
            answers = [6] * 4 if i < 3 else [6 if j == i % 4 else 7 + j for j in range(4)]
            bank.append(dict(id=f'q{i}', question=f'Question {i}: 2 boxes of 3 items.',
                             candidates=[dict(index=j, text=f'#### {a}') for j, a in enumerate(answers)]))
            references.append(dict(id=f'q{i}', question=bank[-1]['question'], answer='#### 6'))
        x = np.zeros((9, 4, 1024), dtype=np.float32)
        seen = []
        def fit(rows, embeddings, labels, variant, **kwargs):
            identifiers = {row['id'] for row in rows}
            self.assertTrue(identifiers)
            self.assertFalse(identifiers & {'q0', 'q1', 'q2'})
            self.assertEqual(kwargs['l2'], .01)
            seen.append((variant, kwargs['drop_shape_features']))
            return {'training_ids': identifiers}, {}
        def predict(rows, embeddings, bundle):
            self.assertFalse({row['id'] for row in rows} & bundle['training_ids'])
            return [int(row['id'][1:]) % 4 for row in rows]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data'
            for name in final.TRAIN_FILES:
                path = data / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())
            config = root / 'config.json'
            config.write_text(json.dumps(final.FINAL_CONFIG), encoding='utf-8')
            with ExitStack() as stack:
                stack.enter_context(patch.object(final, 'RUNS_ROOT', root / 'runs'))
                check = stack.enter_context(patch.object(comparisons, 'check_data'))
                load = stack.enter_context(patch.object(comparisons, 'load_inputs', return_value=(bank, {'embeddings': x})))
                read = stack.enter_context(patch.object(comparisons, 'read_jsonl', return_value=references))
                fitting = stack.enter_context(patch.object(comparisons.group_linear, 'fit', side_effect=fit))
                stack.enter_context(patch.object(comparisons.group_linear, 'predict', side_effect=predict))
                result = comparisons.run_comparisons(config, root / 'runs' / 'cv', data_dir=data)
                check.assert_called_once_with(data.resolve(), ('train',), inputs_only=False)
                load.assert_called_once_with('train', data.resolve())
                read.assert_called_once_with(data.resolve() / 'gsm8k_train.jsonl')
                self.assertEqual(fitting.call_count, 36)
                self.assertEqual(result['fit_count'], 36)
                self.assertEqual(result['informative_questions'], 6)
                self.assertEqual(len(result['comparisons']), 12)
                self.assertEqual(result['comparisons']['group_shape_bt']['correct'], 6)
                self.assertEqual(sorted(p.name for p in (root / 'runs' / 'cv').iterdir()), ['comparisons.json'])
                self.assertEqual(set(seen), {(variant, ()) for variant in comparisons.MAIN_VARIANTS}
                                 | {('group_shape_bt', (feature,)) for feature in comparisons.group_linear.SHAPE_FEATURES})
                with patch.object(comparisons, 'run_comparisons', return_value=result) as run, redirect_stdout(io.StringIO()):
                    self.assertEqual(train.main(['--config', str(config), '--run-dir', str(root / 'runs' / 'new'),
                                                 '--data-dir', str(data), '--reproduce-comparisons']), 0)
                    run.assert_called_once()


if __name__ == '__main__':
    unittest.main()
