# src/train.py
from __future__ import annotations
import argparse, os, joblib, numpy as np, matplotlib.pyplot as plt
from typing import Tuple, Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, classification_report, precision_recall_curve, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from src.utils import ensure_dir, set_seed, dump_json, timestamp

# CatBoost is optional; enable if installed
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False

def load_npz(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    X_train, y_train = d["X_train"], d["y_train"]
    X_val, y_val = d["X_val"], d["y_val"]
    X_test, y_test = d["X_test"], d["y_test"]
    feat_names = d["feat_names"].tolist()
    class_names = d["class_names"].tolist()
    return (X_train, y_train, X_val, y_val, X_test, y_test, feat_names, class_names)

def plot_roc_pr(y_true, y_score, outdir: str, tag: str):
    ensure_dir(outdir)
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUROC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend()
    roc_path = os.path.join(outdir, f"{tag}_roc.png")
    plt.savefig(roc_path, bbox_inches="tight"); plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure()
    plt.plot(rec, prec, label=f"AUPRC={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve"); plt.legend()
    pr_path = os.path.join(outdir, f"{tag}_pr.png")
    plt.savefig(pr_path, bbox_inches="tight"); plt.close()
    return roc_path, pr_path, roc_auc, ap

def plot_confusion(y_true, y_pred, class_names, outdir: str, tag: str):
    ensure_dir(outdir)
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix"); plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    cm_path = os.path.join(outdir, f"{tag}_confusion.png")
    plt.savefig(cm_path, bbox_inches="tight"); plt.close()
    return cm_path

def build_model(name: str, task: str, seed: int,
                iterations: int=500, depth: int=6, learning_rate: float=0.1,
                class_weight_pos: float=1.0, n_jobs: int= -1):
    name = name.lower()
    if name == "catboost":
        if not HAS_CATBOOST:
            raise ImportError("CatBoost not installed. Install catboost or choose another --model.")

        # Respect any values you're already passing; otherwise use safe defaults.
        iters = iterations if (iterations is not None) else 2000
        lr    = learning_rate if (learning_rate is not None) else 0.1

        if task == "classify":
            # Multiclass setup (macro-friendly TotalF1). Depth defaults to 6 unless you provide one.
            dpth = depth if (depth is not None) else 6
            return CatBoostClassifier(
                loss_function="MultiClass",
                eval_metric="TotalF1",
                iterations=iters,
                depth=dpth,
                learning_rate=lr,
                l2_leaf_reg=3,      # small regularization to stabilize multiclass
                random_state=seed,
                verbose=False
            )
        else:
            # Detection (binary) — keep your original settings and class weighting.
            dpth = depth if (depth is not None) else 8
            loss = "Logloss"
            class_weights = [1.0, class_weight_pos]  # your existing positive-class weight
            return CatBoostClassifier(
                iterations=iters,
                depth=dpth,
                learning_rate=lr,
                loss_function=loss,
                random_state=seed,
                eval_metric="AUC",
                class_weights=class_weights,
                verbose=False
            )
    elif name == "logreg":
        # keep existing behavior (uses StandardScaler pipeline)
        return Pipeline([
            ("scaler", StandardScaler(with_mean=True)),
            ("clf", LogisticRegression(
                random_state=seed, max_iter=2000, n_jobs=n_jobs,
                penalty="l2", solver="lbfgs", multi_class="auto"
            ))
        ])
    elif name == "logreg_sparse":
        # For Mal-API TF-IDF / sparse feature matrices
        return LogisticRegression(
            random_state=seed,
            max_iter=2000,
            n_jobs=n_jobs,
            penalty="l2",
            solver="saga",
            multi_class="auto"
        )
    elif name == "rf":
        return RandomForestClassifier(
            n_estimators=400, max_depth=None, random_state=seed, n_jobs=n_jobs,
            class_weight="balanced" if task=="detect" else None
        )
    elif name == "linear_svm":
        # keep as-is for legacy usage (with scaling)
        return Pipeline([
            ("scaler", StandardScaler(with_mean=True)),
            ("clf", LinearSVC(random_state=seed, dual=False))
        ])
    elif name == "linear_svm_calibrated":
        # Linear SVM wrapped in calibration so predict_proba is available for ROC/PR.
        base = LinearSVC(random_state=seed, dual=False, C=1.0)
        return CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=5)
    elif name == "rbf_svm":
        return Pipeline([
            ("scaler", StandardScaler(with_mean=True)),
            ("clf", SVC(kernel="rbf", probability=True, random_state=seed))
        ])
    elif name == "mlp":
        # Small, regularized, with early stopping; good for 240-D dense features
        return MLPClassifier(
            hidden_layer_sizes=(512,),
            activation="relu",
            alpha=1e-4,                # L2
            learning_rate_init=1e-3,
            batch_size=128,
            early_stopping=True,
            n_iter_no_change=10,
            max_iter=200,
            random_state=seed
        )
    else:
        raise ValueError(f"Unknown model '{name}'")

def binary_scores(model, X_test) -> np.ndarray:
    # Prefer predict_proba; else decision_function
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        if isinstance(proba, list):  # catboost multiclass returns list
            proba = np.asarray(proba)
        # binary case: use positive class column
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        if scores.ndim == 1:
            return scores
        if scores.ndim == 2 and scores.shape[1] == 2:
            return scores[:, 1]
    # fallback: use predictions as scores (not ideal)
    return model.predict(X_test).astype(float)

def main():
    parser = argparse.ArgumentParser(description="Train/evaluate malware models")
    parser.add_argument("--dataset_npz", required=True, help="Path produced by preprocess.py")
    parser.add_argument("--task", required=True, choices=["detect", "classify"])
    parser.add_argument("--model", required=True,
                    choices=[
                      "catboost", "logreg", "logreg_sparse",
                      "rf", "linear_svm", "linear_svm_calibrated", "rbf_svm", "mlp"
                    ])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=-1)

    # CatBoost-only hyperparams (ignored by others)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--class_weight_pos", type=float, default=1.0, help="For imbalanced binary task")

    parser.add_argument("--out_models", default="outputs/models")
    parser.add_argument("--out_metrics", default="outputs/metrics")
    parser.add_argument("--out_figs", default="outputs/figures")
    args = parser.parse_args()

    set_seed(args.seed)
    (X_tr, y_tr, X_va, y_va, X_te, y_te, feat_names, class_names) = load_npz(args.dataset_npz)

    ensure_dir(args.out_models); ensure_dir(args.out_metrics); ensure_dir(args.out_figs)
    tag = f"{os.path.splitext(os.path.basename(args.dataset_npz))[0]}_{args.model}_{timestamp()}"

    model = build_model(
        args.model, args.task, args.seed,
        iterations=args.iterations, depth=args.depth, learning_rate=args.learning_rate,
        class_weight_pos=args.class_weight_pos, n_jobs=args.n_jobs
    )

    # Fit with validation set when supported
    if hasattr(model, "fit"):
        try:
            if "CatBoostClassifier" in str(type(model)):
                model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
            else:
                model.fit(np.vstack([X_tr, X_va]), np.concatenate([y_tr, y_va]))
        except TypeError:
            # Some sklearn pipelines don't accept eval_set; just fit on train+val
            model.fit(np.vstack([X_tr, X_va]), np.concatenate([y_tr, y_va]))

    # Evaluate
    if args.task == "detect":
        scores = binary_scores(model, X_te)
        y_pred = model.predict(X_te)
        roc_path, pr_path, auroc, auprc = plot_roc_pr(y_te, scores, args.out_figs, f"{tag}_detect")
        cm_path = plot_confusion(y_te, y_pred, class_names, args.out_figs, f"{tag}_detect")
        f1 = f1_score(y_te, y_pred)
        metrics = {
            "task": "detect",
            "model": args.model,
            "auroc": float(auroc),
            "auprc": float(auprc),
            "f1": float(f1),
            "roc_fig": roc_path,
            "pr_fig": pr_path,
            "cm_fig": cm_path
        }
    else:
        y_pred = model.predict(X_te).astype(int)
        report = classification_report(y_te, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        cm_path = plot_confusion(y_te, y_pred, class_names, args.out_figs, f"{tag}_classify")
        metrics = {
            "task": "classify",
            "model": args.model,
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "weighted_f1": float(report["weighted avg"]["f1-score"]),
            "per_class_f1": {cls: float(report[cls]["f1-score"]) for cls in class_names},
            "cm_fig": cm_path
        }

    # Save artifacts
    model_path = os.path.join(args.out_models, f"{tag}.pkl")
    joblib.dump(model, model_path)
    metrics_path = os.path.join(args.out_metrics, f"{tag}.metrics.json")
    dump_json(metrics, metrics_path)

    print("[OK] Saved model:", model_path)
    print("[OK] Saved metrics:", metrics_path)
    for k in ["roc_fig", "pr_fig", "cm_fig"]:
        if k in metrics:
            print(f"[OK] Figure: {metrics[k]}")

if __name__ == "__main__":
    main()
