import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def best_threshold(y_true, y_prob):
    p, r, t = precision_recall_curve(y_true, y_prob)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    i = int(np.argmax(f1[:-1]))          # f1 has one more entry than t
    return float(t[i]), float(f1[i])


def evaluate(y_true, y_prob, labels):
    out = {}
    for j, name in enumerate(labels):
        thr, f1 = best_threshold(y_true[:, j], y_prob[:, j])
        out[name] = {
            "positives": int(y_true[:, j].sum()),
            "roc_auc": float(roc_auc_score(y_true[:, j], y_prob[:, j])),
            "avg_precision": float(average_precision_score(y_true[:, j], y_prob[:, j])),
            "best_f1": f1,
            "threshold": thr,
        }
    out["macro_roc_auc"] = float(np.mean([out[n]["roc_auc"] for n in labels]))
    out["macro_avg_precision"] = float(np.mean([out[n]["avg_precision"] for n in labels]))
    return out


def print_report(m, labels):
    print(f"{'label':<16}{'pos':>6}{'roc_auc':>9}{'avg_prec':>10}{'best_f1':>9}{'thr':>7}")
    for n in labels:
        r = m[n]
        print(f"{n:<16}{r['positives']:>6}{r['roc_auc']:>9.3f}{r['avg_precision']:>10.3f}"
              f"{r['best_f1']:>9.3f}{r['threshold']:>7.2f}")
    print(f"macro roc_auc={m['macro_roc_auc']:.3f}  macro avg_precision={m['macro_avg_precision']:.3f}")