"""The final training entry fits one fixed model using aligned train labels only."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mp2 import final_training as final
import train


def example(identifier, answers):
    return {'id': identifier, 'question': 'There are 2 boxes with 3 items.',
            'candidates': [{'index': i, 'text': 'No answer' if a is None else f'#### {a}'}
                           for i, a in enumerate(answers)]}


class FinalTrainingTests(unittest.TestCase):
    def setUp(self):
        self.bank = [example('a', ['0.5', '1/2', 2, 3]),
                     example('b', [6, 6, 6, 6]), example('c', [6, None, None, None])]
        self.references = [{'id': row['id'], 'question': row['question'],
                            'answer': '#### 1/2' if row['id'] == 'a' else '#### 6'}
                           for row in self.bank]
        self.x = np.random.default_rng(4).normal(size=(3, 4, 1024)).astype(np.float32)

    def test_exact_numeric_multi_positive_labels_and_alignment(self):
        labels, hard = final.training_labels(self.bank, self.references)
        np.testing.assert_array_equal(labels, [[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]])
        np.testing.assert_array_equal(hard, [True, False, False])
        for references in (self.references[::-1], self.references[:-1],
                           [{**self.references[0], 'question': 'Changed'}, *self.references[1:]],
                           [{**self.references[0], 'answer': '#### 99'}, *self.references[1:]]):
            with self.subTest(references=references), self.assertRaises(ValueError):
                final.training_labels(self.bank, references)

    def test_only_informative_train_rows_fit_and_existing_run_is_refused(self):
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
                check = stack.enter_context(patch.object(final, 'check_data'))
                load = stack.enter_context(patch.object(final, 'load_inputs', return_value=(self.bank, {'embeddings': self.x})))
                read = stack.enter_context(patch.object(final, 'read_jsonl', return_value=self.references))
                fit = stack.enter_context(patch.object(final.group_linear, 'fit', wraps=final.group_linear.fit))
                report = final.train_final(config, root / 'runs' / 'reproduce', data_dir=data)
                check.assert_called_once_with(data.resolve(), ('train',), inputs_only=False)
                load.assert_called_once_with('train', data.resolve())
                read.assert_called_once_with(data.resolve() / 'gsm8k_train.jsonl')
                arguments = fit.call_args
                self.assertEqual(arguments.args[0], self.bank[:1])
                np.testing.assert_array_equal(arguments.args[1], self.x[:1])
                np.testing.assert_array_equal(arguments.args[2], [[1, 1, 0, 0]])
                self.assertEqual(arguments.args[3], 'group_shape_bt')
                self.assertEqual(arguments.kwargs, {'l2': .01})
                self.assertEqual(report['training_support']['disagreement'], 1)
                self.assertTrue(report['diagnostics']['converged'])
                checkpoint = Path(report['model_bundle'])
                self.assertEqual((checkpoint / 'config.json').read_bytes(),
                                 (json.dumps(final.FINAL_CONFIG, indent=2) + '\n').encode())
                self.assertEqual(sorted(p.name for p in checkpoint.iterdir()), ['config.json', 'weights.npz'])
                with self.assertRaises(FileExistsError):
                    final.train_final(config, root / 'runs' / 'reproduce', data_dir=data)
                self.assertEqual(fit.call_count, 1)
                self.assertEqual(check.call_count, 1)

    def test_fixed_configuration_rejects_changes_before_inputs_open(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            bad_configs = [{**final.FINAL_CONFIG, 'l2': .03},
                           {**final.FINAL_CONFIG, 'variant': 'candidate_bt'},
                           {**final.FINAL_CONFIG, 'fold': 0},
                           {**final.FINAL_CONFIG, 'l2': True}, []]
            with patch.object(final, 'check_data') as check, patch.object(final.group_linear, 'fit') as fit:
                for value in bad_configs:
                    config.write_text(json.dumps(value), encoding='utf-8')
                    with self.subTest(config=value), self.assertRaises(ValueError):
                        final.train_final(config, root / 'runs' / 'result')
                check.assert_not_called()
                fit.assert_not_called()

    def test_cli_dispatch_has_no_prepared_data_or_seed_and_rejects_legacy_options(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            config.write_text(json.dumps(final.FINAL_CONFIG), encoding='utf-8')
            base = ['--config', str(config), '--run-dir', str(root / 'out')]
            summary = {'status': 'complete', 'training_support': {'disagreement': 1},
                       'train_seconds': 0.1, 'model_bundle': 'result/model_bundle'}
            with patch.object(final, 'train_final', return_value=summary) as fit:
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(train.main(base + ['--data-dir', str(root / 'data')]), 0)
                self.assertEqual(fit.call_args.args, (config, root / 'out'))
                self.assertEqual(fit.call_args.kwargs['data_dir'], root / 'data')
                for extra in (['--seed', '42'], ['--prepared-dir', str(root / 'prepared')],
                              ['--fold', '0'], ['--fixed-epochs', '20']):
                    with self.subTest(extra=extra), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        train.main(base + extra)
                self.assertEqual(fit.call_count, 1)


if __name__ == '__main__':
    unittest.main()
