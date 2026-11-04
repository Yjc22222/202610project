"""Binary metrics: spoof is the positive class; failures are reported explicitly."""
import numpy as np


def _ratio(top, bottom):
    return float(top / bottom) if bottom else None


def classification_metrics(records):
    total = len(records)
    valid = [r for r in records if r.get('prediction') in ('bonafide', 'spoof') and r['status'] == 'ok']
    positives = sum(r['label'] == 'spoof' for r in records)
    negatives = total - positives
    tp = sum(r['label'] == 'spoof' and r['prediction'] == 'spoof' for r in valid)
    tn = sum(r['label'] == 'bonafide' and r['prediction'] == 'bonafide' for r in valid)
    fp = sum(r['label'] == 'bonafide' and r['prediction'] == 'spoof' for r in valid)
    fn = sum(r['label'] == 'spoof' and r['prediction'] == 'bonafide' for r in valid)
    failed_spoof = positives - tp - fn
    failed_bonafide = negatives - tn - fp
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, positives)
    acceptance = _ratio(tn, negatives)
    f1 = _ratio(2 * tp, 2 * tp + fp + fn + failed_spoof)
    return {
        'total': total, 'scored': len(valid), 'failed': total - len(valid),
        'coverage': _ratio(len(valid), total),
        'accuracy_on_all': _ratio(tp + tn, total),
        'accuracy_on_scored': _ratio(tp + tn, len(valid)),
        'spoof_precision': precision, 'spoof_recall_on_all': recall,
        'spoof_recall_on_scored': _ratio(tp, tp + fn),
        'bonafide_acceptance_on_all': acceptance,
        'bonafide_false_positive_rate_on_scored': _ratio(fp, tn + fp),
        'balanced_accuracy_on_all': (recall + acceptance) / 2 if recall is not None and acceptance is not None else None,
        'spoof_f1_on_all': f1,
        'confusion_matrix_scored': {'TP_spoof': tp, 'TN_bonafide': tn, 'FP_bonafide_as_spoof': fp, 'FN_spoof_as_bonafide': fn},
        'failures_by_label': {'spoof': failed_spoof, 'bonafide': failed_bonafide},
    }


def threshold_curve(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    bona = np.asarray([label == 'bonafide' for label in labels], dtype=np.int64)
    if len(scores) != len(bona) or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Expected matching finite scores and labels.')
    n_bona, n_spoof = int(bona.sum()), int(len(bona) - bona.sum())
    if not n_bona or not n_spoof:
        raise ValueError('Both classes are required to choose a threshold.')
    order = np.argsort(scores, kind='stable')
    values, flags = scores[order], bona[order]
    ends = np.r_[np.flatnonzero(values[1:] != values[:-1]), len(values) - 1]
    thresholds = np.r_[np.nextafter(values[0], -np.inf), values[ends]]
    bona_removed = np.r_[0, np.cumsum(flags)[ends]]
    spoof_removed = np.r_[0, np.cumsum(1 - flags)[ends]]
    false_reject = bona_removed / n_bona
    false_accept = (n_spoof - spoof_removed) / n_spoof
    return thresholds, false_reject, false_accept


def select_threshold(scores, labels, method='balanced_accuracy'):
    thresholds, false_reject, false_accept = threshold_curve(scores, labels)
    error = (false_reject + false_accept) / 2
    gap = np.abs(false_reject - false_accept)
    if method == 'balanced_accuracy':
        selected = int(np.lexsort((gap, error))[0])
    elif method == 'eer':
        selected = int(np.lexsort((error, gap))[0])
    else:
        raise ValueError('Unknown threshold selection method.')
    eer_index = int(np.lexsort((error, gap))[0])
    return {'threshold': float(thresholds[selected]), 'method': method,
            'validation_balanced_accuracy': float(1 - error[selected]),
            'validation_eer': float(error[eer_index]),
            'validation_bonafide_false_reject_rate': float(false_reject[selected]),
            'validation_spoof_false_accept_rate': float(false_accept[selected])}
