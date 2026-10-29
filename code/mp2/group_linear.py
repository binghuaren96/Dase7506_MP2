"""Full-dimensional linear Bradley--Terry scorers for E17/E18.

Only ``fit`` imports SciPy. Frozen forward paths use NumPy, supplied texts and
embeddings; identifiers, references and cached predictions are never inputs.
All objective weights are normalized per original question and wrong answer.
"""
from fractions import Fraction
import math
import re

import numpy as np

from common import parse_answer
from mp2.selection import build_groups

VARIANTS = ('candidate_bt', 'group_bt', 'group_shape_bt', 'candidate_shape_bt')
INPUT_DIM = 1024
SHAPE_FEATURES = ('support_fraction', 'log_variance', 'mean_pair_cosine',
                  'signed_log_abs', 'integer', 'zero', 'negative',
                  'exact_in_question', 'signed_log_relative_question_max')
NUMBER = re.compile(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?')
BUNDLE_KEYS = frozenset(('variant', 'mean', 'scale', 'coefficient', 'input_dim'))


def _embeddings(bank, embeddings):
    x = np.asarray(embeddings)
    if (x.dtype.kind not in 'fiu' or x.shape != (len(bank), 4, INPUT_DIM)
            or not np.isfinite(x).all()):
        raise ValueError('Expected finite real embeddings [N,4,1024] aligned to bank')
    return x.astype(np.float64, copy=False)


def _variant(variant):
    if variant not in VARIANTS:
        raise ValueError('Unsupported E17 linear variant')
    return variant


def _numbers(question):
    values = set()
    for token in NUMBER.findall(str(question)[:20000])[:512]:
        try:
            values.add(Fraction(token.replace(',', '')))
        except (ValueError, ZeroDivisionError):
            pass
    return values


def _signed_log_abs(value):
    """Compute sign(x)*log(1+abs(x)) without float(Fraction) overflow."""
    if value == 0:
        return 0.0
    logarithm = math.log(abs(value.numerator)) - math.log(value.denominator)
    result = logarithm if logarithm > 40 else math.log1p(math.exp(logarithm))
    return result if value > 0 else -result


def build_features(bank, embeddings, variant):
    """Return finite features, active actions, and canonical answer membership.

    Candidate variants use original candidate slots; group variants use
    first-occurrence numeric-answer groups padded to four actions. No position
    is itself a learned feature. All-invalid input is legal for inference.
    """
    _variant(variant)
    x = _embeddings(bank, embeddings)
    membership = build_groups(bank)
    if variant == 'candidate_bt':
        return dict(features=x.copy(), active=membership.any(axis=1), membership=membership)
    counts = membership.sum(axis=2)
    active = counts > 0
    means = np.einsum('ngc,ncd->ngd', membership.astype(np.float64), x)
    means /= np.maximum(counts[..., None], 1)
    if variant == 'group_bt':
        return dict(features=means, active=active, membership=membership)
    shape = np.zeros((len(bank), 4, len(SHAPE_FEATURES)), dtype=np.float64)
    for q, row in enumerate(bank):
        question_numbers = _numbers(row['question'])
        maximum = max((abs(v) for v in question_numbers), default=Fraction(0))
        for g in np.flatnonzero(active[q]):
            indices = np.flatnonzero(membership[q, g])
            values = x[q, indices]
            variance = float(np.square(values - means[q, g]).mean())
            cosine = 0.0
            if len(indices) > 1:
                units = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
                pair = (units @ units.T)[np.triu_indices(len(indices), 1)]
                cosine = float(np.clip(pair, -1, 1).mean())
            answer = Fraction(parse_answer(row['candidates'][int(indices[0])]['text']))
            shape[q, g] = [len(indices) / 4, math.log1p(variance), cosine,
                _signed_log_abs(answer), answer.denominator == 1, answer == 0, answer < 0,
                answer in question_numbers, _signed_log_abs(answer / maximum) if maximum else 0]
    if variant == 'candidate_shape_bt':
        # Preserve each trace's embedding; every member shares its group's
        # statistics. Keep the unique groups for training-only standardization
        # so a group with three members does not count three times there.
        candidate_shape = np.einsum('ngc,ngd->ncd', membership.astype(np.float64), shape)
        features = np.concatenate((x, candidate_shape), axis=2)
        if not np.isfinite(features).all():
            raise ValueError('Nonfinite candidate shape features')
        return dict(features=features, active=membership.any(axis=1), membership=membership,
                    group_shape=shape, group_active=active)
    features = np.concatenate((means, shape), axis=2)
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite group shape features')
    return dict(features=features, active=active, membership=membership)


def _labels(feature_bank, labels):
    membership = feature_bank['membership']
    n = len(membership)
    y = np.asarray(labels)
    if y.dtype != np.bool_ or y.shape != (n, 4) or not y.any(axis=1).all():
        raise ValueError('Expected aligned boolean labels with at least one correct candidate')
    if (y & ~membership.any(axis=1)).any():
        raise ValueError('Correct candidates must have parseable numeric answers')
    count = membership.sum(axis=2)
    positive_count = (membership & y[:, None, :]).sum(axis=2)
    if ((positive_count > 0) & (positive_count != count)).any():
        raise ValueError('Same numeric answer cannot have inconsistent targets')
    positive_group = positive_count > 0
    if not (positive_group.sum(axis=1) == 1).all():
        raise ValueError('There must be exactly one correct numeric answer group')
    return y, positive_group


def pair_data(feature_bank, labels, variant):
    """Build raw feature differences with equal question/wrong-group weights.

    All-correct/no-valid-disagreement rows contribute no pairs. Candidate
    pairs split wrong-group weight across its repeated candidates and across
    repeated positive candidates. Group variants have one pair/wrong group.
    """
    _variant(variant)
    y, positive_group = _labels(feature_bank, labels)
    x = np.asarray(feature_bank['features'], dtype=np.float64)
    membership = feature_bank['membership']
    valid_group = membership.any(axis=2)
    informative = valid_group.sum(axis=1) >= 2
    differences, weights, questions, winners, losers = [], [], [], [], []
    for q in np.flatnonzero(informative):
        good_group = int(np.argmax(positive_group[q]))
        bad_groups = np.flatnonzero(valid_group[q] & ~positive_group[q])
        candidate_actions = variant in ('candidate_bt', 'candidate_shape_bt')
        good = np.flatnonzero(membership[q, good_group]) if candidate_actions else [good_group]
        for bad_group in bad_groups:
            bad = np.flatnonzero(membership[q, bad_group]) if candidate_actions else [int(bad_group)]
            weight = 1 / (len(bad_groups) * len(good) * len(bad))
            for i in good:
                for j in bad:
                    differences.append(x[q, i] - x[q, j])
                    weights.append(weight);questions.append(q);winners.append(i);losers.append(j)
    d = x.shape[-1]
    return dict(differences=np.asarray(differences, dtype=np.float64).reshape(-1, d),
                weights=np.asarray(weights, dtype=np.float64), questions=np.asarray(questions, dtype=np.int64),
                winners=np.asarray(winners, dtype=np.int64), losers=np.asarray(losers, dtype=np.int64),
                informative=informative)


def objective_gradient(coefficient, differences, weights, l2):
    """Weighted mean softplus(-margin) plus (l2/2)*||coefficient||^2."""
    w = np.asarray(coefficient, dtype=np.float64)
    delta = np.asarray(differences, dtype=np.float64)
    mass = np.asarray(weights, dtype=np.float64)
    if (delta.ndim != 2 or w.shape != (delta.shape[1],) or mass.shape != (len(delta),)
            or not len(delta) or not np.isfinite(delta).all() or not np.isfinite(w).all()
            or not np.isfinite(mass).all() or (mass <= 0).any()
            or not np.isfinite(l2) or l2 < 0):
        raise ValueError('Expected finite differences, positive weights and nonnegative L2')
    margin = delta @ w
    normalizer = mass.sum()
    loss = float(mass @ np.logaddexp(0, -margin) / normalizer + .5 * l2 * (w @ w))
    derivative = -mass * np.exp(-np.logaddexp(0, margin)) / normalizer
    gradient = delta.T @ derivative + l2 * w
    return loss, gradient


def fit(bank, embeddings, labels, variant, l2=.01, drop_shape_features=()):
    """Fit on supplied training fold only; caller must not pass validation.

    The embedding block uses the same original four-candidate statistics for
    all variants, fitted only on informative questions. Group pooling is
    affine, so pooling before this transform equals pooling afterward. Only
    the appended shape block uses its own valid-group statistics. For group
    shape ablations, named columns are zeroed in the training differences
    before refitting. The full standardizer and frozen asset schema remain
    unchanged; deleted columns consequently have exactly zero coefficients.
    """
    from scipy.optimize import minimize
    if isinstance(drop_shape_features, (str, bytes)):
        raise ValueError('Dropped shape features must be an iterable of unique feature names')
    try:
        dropped = tuple(drop_shape_features)
    except TypeError as exc:
        raise ValueError('Dropped shape features must be an iterable of unique feature names') from exc
    if dropped:
        if variant != 'group_shape_bt':
            raise ValueError('Shape-feature ablation requires group_shape_bt')
        if (any(not isinstance(name, str) or name not in SHAPE_FEATURES for name in dropped)
                or len(set(dropped)) != len(dropped)):
            raise ValueError('Dropped shape features must be known, unique feature names')
        dropped = tuple(name for name in SHAPE_FEATURES if name in dropped)
        dropped_columns = np.asarray([INPUT_DIM + SHAPE_FEATURES.index(name) for name in dropped], dtype=np.int64)
    built = build_features(bank, embeddings, variant)
    pairs = pair_data(built, labels, variant)
    if not len(pairs['weights']):
        raise ValueError('At least one valid answer-disagreement training question is required')
    selected = built['active'] & pairs['informative'][:, None]
    raw = _embeddings(bank, embeddings)[pairs['informative']].reshape(-1, INPUT_DIM)
    mean = raw.mean(axis=0)
    scale = np.maximum(raw.std(axis=0), 1e-6)
    if variant in ('group_shape_bt', 'candidate_shape_bt'):
        if variant == 'candidate_shape_bt':
            group_selected = built['group_active'] & pairs['informative'][:, None]
            shape = built['group_shape'][group_selected]
        else:
            shape = built['features'][selected][:, INPUT_DIM:]
        mean = np.concatenate((mean, shape.mean(axis=0)))
        scale = np.concatenate((scale, np.maximum(shape.std(axis=0), 1e-6)))
    delta = pairs['differences'] / scale
    if dropped:
        delta[:, dropped_columns] = 0.0
    optimized = minimize(objective_gradient, np.zeros(len(scale)),
        args=(delta, pairs['weights'], l2), method='L-BFGS-B', jac=True,
        options=dict(maxiter=500, ftol=1e-10, gtol=1e-7, maxls=50))
    if not optimized.success or not np.isfinite(optimized.x).all():
        raise RuntimeError('Fixed L-BFGS optimization failed: ' + str(optimized.message))
    if dropped and np.any(optimized.x[dropped_columns] != 0):
        raise RuntimeError('Ablated training columns must retain exactly zero coefficients')
    bundle = dict(variant=np.asarray(variant), mean=mean, scale=scale,
                  coefficient=optimized.x.astype(np.float64), input_dim=np.asarray(INPUT_DIM, dtype=np.int64))
    report = dict(variant=variant, converged=True, iterations=int(optimized.nit),
                  function_evaluations=int(optimized.nfev), loss=float(optimized.fun),
                  gradient_max_abs=float(np.abs(optimized.jac).max()), l2=float(l2),
                  informative_questions=int(pairs['informative'].sum()),
                  standardizer_actions=len(raw), valid_feature_actions=int(selected.sum()), pairs=len(pairs['weights']),
                  pair_weight_sum=float(pairs['weights'].sum()), feature_dimension=len(scale),
                  optimizer='L-BFGS-B', max_iterations=500,
                  standardizer_scope='all_four_raw_candidates_of_supplied_informative_train_questions_only',
                  shape_standardizer_scope='valid_groups_of_supplied_informative_train_questions_only')
    if variant == 'candidate_shape_bt':
        report['shape_standardizer_actions'] = len(shape)
    if dropped:
        report.update(drop_feature_names=list(dropped), zeroed_coefficient_indices=dropped_columns.tolist(),
                      effective_feature_dimension=len(scale)-len(dropped),
                      ablation_scope='training_differences_zeroed_before_refitting')
    return bundle, report


def _bundle(bundle):
    if set(bundle) != BUNDLE_KEYS:
        raise ValueError('Unexpected frozen linear bundle keys')
    variant_array = np.asarray(bundle['variant'])
    if variant_array.shape != () or variant_array.dtype.kind not in 'US':
        raise ValueError('Frozen variant must be a scalar string')
    variant = _variant(str(variant_array.item()))
    if np.asarray(bundle['input_dim']).shape != () or int(bundle['input_dim']) != INPUT_DIM:
        raise ValueError('Frozen input dimension must be 1024')
    d = INPUT_DIM + (len(SHAPE_FEATURES) if variant in ('group_shape_bt', 'candidate_shape_bt') else 0)
    converted = {}
    for key in ('mean', 'scale', 'coefficient'):
        value = np.asarray(bundle[key])
        if value.dtype.kind not in 'fiu' or value.shape != (d,) or not np.isfinite(value).all():
            raise ValueError('Invalid frozen ' + key)
        converted[key] = value.astype(np.float64, copy=False)
    if (converted['scale'] <= 0).any():
        raise ValueError('Frozen scale must be positive')
    return variant, converted


def scores(bank, embeddings, bundle):
    """Return action logits, -inf only for excluded/empty actions.

    On an all-invalid row candidate variants expose finite fallback scores; group
    variants retain four -inf padded groups and ``predict`` supplies fallback.
    """
    variant, frozen = _bundle(bundle)
    built = build_features(bank, embeddings, variant)
    logits = ((built['features'] - frozen['mean']) / frozen['scale']) @ frozen['coefficient']
    active = built['active']
    if variant in ('candidate_bt', 'candidate_shape_bt'):
        active = active | ~active.any(axis=1, keepdims=True)
    return np.where(active, logits, -np.inf)


def predict(bank, embeddings, bundle):
    """Select a candidate, using first group/member for exact group ties."""
    variant, frozen = _bundle(bundle)
    logits = scores(bank, embeddings, bundle)
    if variant in ('candidate_bt', 'candidate_shape_bt'):
        return np.argmax(logits, axis=1).tolist()
    membership = build_groups(bank)
    result = []
    x = _embeddings(bank, embeddings)
    fallback = ((x - frozen['mean'][:INPUT_DIM]) / frozen['scale'][:INPUT_DIM]) @ frozen['coefficient'][:INPUT_DIM]
    for q, row in enumerate(logits):
        if not np.isfinite(row).any():
            result.append(int(np.argmax(fallback[q])))
        else:
            group = int(np.argmax(row))
            result.append(int(np.flatnonzero(membership[q, group])[0]))
    return result
