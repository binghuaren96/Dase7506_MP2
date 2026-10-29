"""E17/E18 objective, weighting, representation and inference contracts."""
from fractions import Fraction
from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import parse_answer
from mp2 import group_linear as linear


def row(answers, question='There are 2 boxes of 3 items, totaling 6 items.'):
    return dict(question=question, candidates=[dict(index=i, text=f'Working. #### {a}')
                if a is not None else dict(index=i, text='No parsed final answer') for i,a in enumerate(answers)])


class GroupLinearTests(unittest.TestCase):
    def setUp(self):
        self.bank = [row([6,6,7,8]), row([6,7,7,7]), row([6,6,6,6]), row([6,None,None,None])]
        self.x = np.random.default_rng(110).normal(size=(4,4,1024)).astype(np.float32)
        self.y = np.array([[1,1,0,0],[1,0,0,0],[1,1,1,1],[1,0,0,0]], dtype=bool)

    def test_gradient_finite_difference_and_extreme_margins(self):
        rng = np.random.default_rng(14)
        delta = rng.normal(size=(9,5)); weight = rng.uniform(.1,2,size=9); coefficient = rng.normal(size=5)
        loss,gradient = linear.objective_gradient(coefficient, delta, weight, .03)
        self.assertTrue(np.isfinite(loss))
        epsilon = 1e-6
        for j in range(5):
            step = np.eye(5)[j] * epsilon
            forward = linear.objective_gradient(coefficient+step, delta, weight, .03)[0]
            backward = linear.objective_gradient(coefficient-step, delta, weight, .03)[0]
            self.assertAlmostEqual((forward-backward)/(2*epsilon), gradient[j], places=7)
        value,grad = linear.objective_gradient(np.ones(2),np.array([[1000,1000],[-1000,-1000]]),np.ones(2),.01)
        self.assertTrue(np.isfinite(value));self.assertTrue(np.isfinite(grad).all())

    def test_empty_ablation_preserves_default_bundle_and_report_exactly(self):
        original, report = linear.fit(self.bank, self.x, self.y, 'group_shape_bt', l2=.01)
        explicit, explicit_report = linear.fit(self.bank, self.x, self.y, 'group_shape_bt',
                                             l2=.01, drop_shape_features=[])
        self.assertEqual(report, explicit_report)
        self.assertNotIn('drop_feature_names', report)
        self.assertNotIn('zeroed_coefficient_indices', report)
        for key in original:
            np.testing.assert_array_equal(original[key], explicit[key])

    def test_ablation_refits_remaining_features_and_matches_training_objective(self):
        deleted = ('signed_log_abs', 'support_fraction')
        columns = [linear.INPUT_DIM + linear.SHAPE_FEATURES.index(name) for name in deleted]
        original, _ = linear.fit(self.bank, self.x, self.y, 'group_shape_bt', l2=.01)
        ablated, report = linear.fit(self.bank, self.x, self.y, 'group_shape_bt',
                                    l2=.01, drop_shape_features=deleted)
        for key in ('mean', 'scale'):
            np.testing.assert_array_equal(ablated[key], original[key])
        self.assertEqual(set(ablated), linear.BUNDLE_KEYS)
        np.testing.assert_array_equal(ablated['coefficient'][columns], 0)
        self.assertEqual(report['drop_feature_names'], ['support_fraction', 'signed_log_abs'])
        self.assertEqual(report['zeroed_coefficient_indices'], sorted(columns))
        self.assertEqual(report['effective_feature_dimension'], 1031)
        self.assertEqual(report['feature_dimension'], 1033)
        built = linear.build_features(self.bank, self.x, 'group_shape_bt')
        pair = linear.pair_data(built, self.y, 'group_shape_bt')
        delta = pair['differences'] / original['scale']
        delta[:, columns] = 0
        loss, gradient = linear.objective_gradient(ablated['coefficient'], delta, pair['weights'], .01)
        self.assertAlmostEqual(loss, report['loss'], places=12)
        self.assertAlmostEqual(float(abs(gradient).max()), report['gradient_max_abs'], places=12)
        self.assertLess(float(abs(gradient).max()), 1e-5)
        truncated = original['coefficient'].copy()
        truncated[columns] = 0
        truncated_loss = linear.objective_gradient(truncated, delta, pair['weights'], .01)[0]
        self.assertLess(loss, truncated_loss-1e-9)
        keep = np.ones(1033, dtype=bool);keep[columns] = False
        self.assertGreater(np.linalg.norm(ablated['coefficient'][keep]-truncated[keep]), 1e-6)
        # The old inference interface needs only the refitted zero coefficients.
        from method import predict
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint/'config.json').write_text(json.dumps(dict(method='group_linear_bt', variant='group_shape_bt')))
            np.savez(checkpoint/'weights.npz', **ablated)
            choices = linear.predict(self.bank, self.x, ablated)
            self.assertEqual(predict(self.bank, dict(embeddings=self.x), checkpoint), choices)
        perm = [2,0,3,1]
        permuted = [dict(question=r['question'], candidates=[r['candidates'][i] for i in perm]) for r in self.bank]
        new_choices = linear.predict(permuted, self.x[:,perm], ablated)
        # Equivalent-answer members use the first member, so compare numeric answers.
        self.assertEqual([parse_answer(r['candidates'][c]['text']) for r,c in zip(self.bank, choices)],
                         [parse_answer(r['candidates'][c]['text']) for r,c in zip(permuted, new_choices)])

    def test_dropping_entire_shape_block_recovers_embedding_only_group_model(self):
        ablated, _ = linear.fit(self.bank, self.x, self.y, 'group_shape_bt', l2=.01,
                                drop_shape_features=linear.SHAPE_FEATURES)
        group, _ = linear.fit(self.bank, self.x, self.y, 'group_bt', l2=.01)
        np.testing.assert_array_equal(ablated['coefficient'][1024:], 0)
        for key in ('mean', 'scale'):
            np.testing.assert_array_equal(ablated[key][:1024], group[key])
        np.testing.assert_allclose(ablated['coefficient'][:1024], group['coefficient'], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(linear.scores(self.bank, self.x, ablated),
                                   linear.scores(self.bank, self.x, group), rtol=1e-10, atol=1e-12)

    def test_ablation_mask_rejects_bad_names_duplicates_and_other_variants(self):
        for deleted in (None, 1, 'zero', ('missing',), ('zero','zero'), (None,), (['zero'],)):
            with self.subTest(deleted=str(deleted)):
                with self.assertRaises(ValueError):
                    linear.fit(self.bank, self.x, self.y, 'group_shape_bt', drop_shape_features=deleted)
        for variant in ('candidate_bt', 'group_bt', 'candidate_shape_bt'):
            with self.subTest(variant=variant):
                with self.assertRaises(ValueError):
                    linear.fit(self.bank, self.x, self.y, variant, drop_shape_features=['zero'])

    def test_equal_question_wrong_group_and_duplicate_weights(self):
        for variant in linear.VARIANTS:
            features = linear.build_features(self.bank,self.x,variant)
            pairs = linear.pair_data(features,self.y,variant)
            np.testing.assert_array_equal(pairs['informative'],[True,True,False,False])
            np.testing.assert_allclose(np.bincount(pairs['questions'],weights=pairs['weights']),1)
            if variant in ('candidate_bt', 'candidate_shape_bt'):
                first = pairs['questions']==0
                self.assertAlmostEqual(pairs['weights'][first & (pairs['losers']==2)].sum(),.5)
                self.assertAlmostEqual(pairs['weights'][first & (pairs['losers']==3)].sum(),.5)
                np.testing.assert_allclose(pairs['weights'][pairs['questions']==1],1/3)
            else:
                np.testing.assert_allclose(pairs['weights'][pairs['questions']==0],.5)
                np.testing.assert_allclose(pairs['weights'][pairs['questions']==1],1)

    def test_shape_mean_exact_answer_and_question_features(self):
        bank = [row(['0.5','1/2',2,-3], question='Half is 0.5, with 2 bags.')]
        features = linear.build_features(bank,self.x[:1],'group_shape_bt')
        np.testing.assert_allclose(features['features'][0,0,:1024],self.x[0,:2].astype(np.float64).mean(axis=0))
        shape = features['features'][0,0,1024:]
        self.assertEqual(len(shape),len(linear.SHAPE_FEATURES))
        self.assertEqual(shape[0],.5);self.assertEqual(shape[4],0);self.assertEqual(shape[7],1)
        self.assertAlmostEqual(shape[3],np.log1p(.5))
        self.assertAlmostEqual(shape[8],np.log1p(.25))
        zero_x = np.zeros_like(self.x[:1])
        self.assertTrue(np.isfinite(linear.build_features(bank,zero_x,'group_shape_bt')['features']).all())

    def test_common_center_cancels_candidate_and_group_margins(self):
        for variant in ('candidate_bt','group_bt'):
            original = linear.build_features(self.bank,self.x,variant)
            centered = linear.build_features(self.bank,self.x-self.x.mean(axis=1,keepdims=True),variant)
            p = linear.pair_data(original,self.y,variant)
            c = linear.pair_data(centered,self.y,variant)
            np.testing.assert_allclose(p['differences'],c['differences'],atol=4e-7)

    def test_candidate_shape_preserves_traces_and_copies_equivalent_group_features(self):
        bank = [row(['0.5','1/2',2,None], question='There are 0.5 liters and 2 cups.'),
                row([6,7,7,7])]
        candidate = linear.build_features(bank,self.x[:2],'candidate_shape_bt')
        group = linear.build_features(bank,self.x[:2],'group_shape_bt')
        np.testing.assert_array_equal(candidate['features'][...,:1024],self.x[:2])
        np.testing.assert_array_equal(candidate['active'],[[True,True,True,False],[True]*4])
        for q in range(len(bank)):
            for g in np.flatnonzero(group['active'][q]):
                for c in np.flatnonzero(group['membership'][q,g]):
                    np.testing.assert_array_equal(candidate['features'][q,c,1024:],group['features'][q,g,1024:])
        np.testing.assert_array_equal(candidate['features'][0,3,1024:],np.zeros(9))
        # Equivalent numeric answers share group features despite distinct text.
        np.testing.assert_array_equal(candidate['features'][0,0,1024:],candidate['features'][0,1,1024:])
        y = np.array([[True,True,False,False],[True,False,False,False]])
        raw_pairs = linear.pair_data(linear.build_features(bank,self.x[:2],'candidate_bt'),y,'candidate_bt')
        shape_pairs = linear.pair_data(candidate,y,'candidate_shape_bt')
        for key in ('weights','questions','winners','losers','informative'):
            np.testing.assert_array_equal(shape_pairs[key],raw_pairs[key])
        np.testing.assert_array_equal(shape_pairs['differences'][:,:1024],raw_pairs['differences'])

    def test_candidate_shape_uses_unique_group_standardization_and_unified_roundtrip(self):
        bank = [row(['0.5','1/2',2,None]),row([6,7,7,7])]
        y = np.array([[True,True,False,False],[True,False,False,False]])
        group = linear.build_features(bank,self.x[:2],'group_shape_bt')
        built = linear.build_features(bank,self.x[:2],'candidate_shape_bt')
        group_values = group['features'][group['active']][:,1024:]
        candidate_values = built['features'][built['active']][:,1024:]
        expected_mean = group_values.mean(axis=0)
        expected_scale = np.maximum(group_values.std(axis=0),1e-6)
        self.assertFalse(np.allclose(expected_mean,candidate_values.mean(axis=0)))
        candidate_bundle,report = linear.fit(bank,self.x[:2],y,'candidate_shape_bt')
        group_bundle,_ = linear.fit(bank,self.x[:2],y,'group_shape_bt')
        self.assertEqual(report['shape_standardizer_actions'],4)
        np.testing.assert_array_equal(candidate_bundle['mean'][1024:],expected_mean)
        np.testing.assert_array_equal(candidate_bundle['scale'][1024:],expected_scale)
        for key in ('mean','scale'):
            np.testing.assert_array_equal(candidate_bundle[key],group_bundle[key])
        from method import predict
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint/'config.json').write_text(json.dumps(dict(method='group_linear_bt',variant='candidate_shape_bt')))
            np.savez(checkpoint/'weights.npz',**candidate_bundle)
            self.assertEqual(predict(bank,dict(embeddings=self.x[:2]),checkpoint),
                             linear.predict(bank,self.x[:2],candidate_bundle))

    def test_candidate_shape_permutation_with_group_evidence(self):
        # Nonzero shape coefficients exercise the membership remapping, while
        # unique semantic scores avoid an order-dependent exact tie.
        bank = [row(['0.5','1/2',2,None]),row([None]*4)]
        x = self.x[:2]
        coefficient = np.random.default_rng(40).normal(size=1033)
        bundle = dict(variant=np.asarray('candidate_shape_bt'),input_dim=np.asarray(1024),
                      mean=np.zeros(1033),scale=np.ones(1033),coefficient=coefficient)
        choices = linear.predict(bank,x,bundle)
        original_scores = linear.scores(bank,x,bundle)
        perm = [2,0,3,1]
        permuted = [dict(question=r['question'],candidates=[r['candidates'][i] for i in perm]) for r in bank]
        new_scores = linear.scores(permuted,x[:,perm],bundle)
        np.testing.assert_allclose(new_scores,original_scores[:,perm],rtol=1e-13,atol=1e-13)
        self.assertEqual([perm[i] for i in linear.predict(permuted,x[:,perm],bundle)],choices)
        self.assertTrue(np.isneginf(original_scores[0,3]))
        self.assertTrue(np.isfinite(original_scores[1]).all())
        self.assertEqual(choices[1],int(np.argmax(x[1]@coefficient[:1024])))

    def test_fit_shared_raw_standardizer_and_roundtrip(self):
        # Easy/no-disagreement rows must never influence embedding statistics.
        x = self.x.astype(np.float64).copy();x[2]+=1000;x[3]-=1000
        expected = x[:2].reshape(-1,1024)
        expected_mean,expected_scale = expected.mean(axis=0),np.maximum(expected.std(axis=0),1e-6)
        for variant in linear.VARIANTS:
            bundle,report = linear.fit(self.bank,x,self.y,variant,l2=.01)
            self.assertTrue(report['converged']);self.assertEqual(report['informative_questions'],2)
            self.assertEqual(report['standardizer_actions'],8)
            np.testing.assert_array_equal(bundle['mean'][:1024],expected_mean)
            np.testing.assert_array_equal(bundle['scale'][:1024],expected_scale)
            old_scores = linear.scores(self.bank,x,bundle)
            archive_bytes = BytesIO();np.savez(archive_bytes,**bundle);archive_bytes.seek(0)
            with np.load(archive_bytes,allow_pickle=False) as archive:
                loaded = {k:archive[k] for k in archive.files}
            np.testing.assert_array_equal(linear.scores(self.bank,x,loaded),old_scores)
            self.assertEqual(linear.predict(self.bank,x,loaded),linear.predict(self.bank,x,bundle))

    def test_permutation_and_all_invalid_fallback(self):
        coefficient = np.zeros(1024);coefficient[0]=1
        for variant in linear.VARIANTS:
            d = 1024+(len(linear.SHAPE_FEATURES) if variant in ('group_shape_bt','candidate_shape_bt') else 0)
            bundle = dict(variant=np.asarray(variant),input_dim=np.asarray(1024),mean=np.zeros(d),
                          scale=np.ones(d),coefficient=np.pad(coefficient,(0,d-1024)))
            bank = [row([6,6,7,None]),row([None]*4)]
            x = np.zeros((2,4,1024));x[0,:,0]=[1,2,8,20];x[1,:,0]=[1,4,2,3]
            choices = linear.predict(bank,x,bundle)
            self.assertEqual(choices,[2,1])
            perm = [2,0,3,1]
            permuted = [dict(question=r['question'],id='unused',candidates=[r['candidates'][i] for i in perm]) for r in bank]
            new = linear.predict(permuted,x[:,perm],bundle)
            self.assertEqual([perm[i] for i in new],choices)

    def test_invalid_inputs_and_no_informative_data(self):
        with self.assertRaises(ValueError):linear.fit(self.bank[2:],self.x[2:],self.y[2:],'group_bt')
        with self.assertRaises(ValueError):linear.build_features(self.bank,self.x[:,:,:2],'group_bt')
        wrong = self.y.copy();wrong[0,1]=False
        with self.assertRaises(ValueError):linear.pair_data(linear.build_features(self.bank,self.x,'candidate_bt'),wrong,'candidate_bt')
        self.assertAlmostEqual(linear._signed_log_abs(Fraction(10**1000)),1000*np.log(10),places=10)


if __name__=='__main__':unittest.main()
