# src/preprocess.py
from __future__ import annotations
import argparse, os, numpy as np
from sklearn.model_selection import train_test_split
from src.utils import ensure_dir, set_seed, dump_json, timestamp
from src.dataio import load_ember2018, load_malapi2019, load_bodmas, load_big2015

def _save_npz_bundle(out_npz: str,
                     X_tr, y_tr, X_va, y_va, X_te, y_te,
                     feat_names, class_names) -> None:
    ensure_dir(os.path.dirname(out_npz))
    np.savez_compressed(
        out_npz,
        X_train=X_tr, y_train=y_tr,
        X_val=X_va, y_val=y_va,
        X_test=X_te, y_test=y_te,
        feat_names=np.array(feat_names, dtype=object),
        class_names=np.array(class_names, dtype=object)
    )

def main():
    p = argparse.ArgumentParser(description="Preprocess malware datasets → canonical NPZ snapshot")
    p.add_argument("--dataset", required=True, choices=["ember2018", "malapi2019", "bodmas", "big2015"])
    p.add_argument("--task", required=True, choices=["detect", "classify"])

    # I/O and layout
    p.add_argument("--data_dir", required=True, help="Root dir for the dataset (contains features.* or shards)")
    p.add_argument("--features_file", default=None, help="Optional explicit path to features.(npz|parquet|csv)")
    p.add_argument("--label_col", default=None, help="Override label column name for table files")
    p.add_argument("--id_col", default=None, help="Optional ID column to drop")
    p.add_argument("--drop_cols", nargs="*", default=None, help="Extra columns to drop (e.g., path, sha256)")

    # Class filtering (multiclass)
    p.add_argument("--top_k", type=int, default=None, help="Keep top-K most frequent classes (multiclass only)")
    p.add_argument("--min_per_class", type=int, default=1, help="Min samples required to keep a class (multiclass)")

    # Splits and sampling
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument("--limit", type=int, default=None, help="Row cap for quick runs")
    p.add_argument("--seed", type=int, default=42)

    # Output
    p.add_argument("--outdir", default="outputs/metrics/datasets")
    args = p.parse_args()

    set_seed(args.seed)

    if args.dataset == "ember2018":
        # Detection only
        if args.task != "detect":
            raise ValueError("EMBER2018 path here is for binary detection. Use --task detect.")
        X, y, feat_names, class_names = load_ember2018(
            data_dir=args.data_dir,
            features_file=args.features_file,
            label_col=args.label_col or "label",
            id_col=args.id_col,
            drop_cols=args.drop_cols,
            limit=args.limit,
        )

    elif args.dataset == "malapi2019":
        # Family classification (API-based)
        if args.task != "classify":
            raise ValueError("MalAPI2019 is wired as a multiclass family classification dataset. Use --task classify.")
        X, y, feat_names, class_names = load_malapi2019(
            data_dir=args.data_dir,
            features_file=args.features_file,
            label_col=args.label_col or "family",
            id_col=args.id_col,
            drop_cols=args.drop_cols,
            top_k=args.top_k,
            min_per_class=args.min_per_class,
            limit=args.limit,
        )

    elif args.dataset == "bodmas":
        # Family classification (PE/static)
        if args.task != "classify":
            raise ValueError("BODMAS is wired as a multiclass family classification dataset. Use --task classify.")
        X, y, feat_names, class_names = load_bodmas(
            data_dir=args.data_dir,
            features_file=args.features_file,
            label_col=args.label_col or "family",  # override via --label_col if needed
            id_col=args.id_col,
            drop_cols=args.drop_cols,
            top_k=args.top_k,
            min_per_class=args.min_per_class,
            limit=args.limit,
        )

    elif args.dataset == "big2015":
        # Family classification (byte-histogram-based, from big2015_extractor.py NPZ)
        if args.task != "classify":
            raise ValueError("BIG2015 is wired as a multiclass family classification dataset. Use --task classify.")
        X, y, feat_names, class_names = load_big2015(
            data_dir=args.data_dir,
            features_file=args.features_file,
            limit=args.limit,
        )

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Train/Val/Test stratified splits
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=args.val_size, stratify=y_tr, random_state=args.seed
    )

    tag = f"{args.dataset}_{args.task}_{timestamp()}"
    out_npz = os.path.join(args.outdir, f"{tag}.npz")
    _save_npz_bundle(out_npz, X_tr, y_tr, X_va, y_va, X_te, y_te, feat_names, class_names)

    meta = {
        "dataset": args.dataset,
        "task": args.task,
        "npz_path": out_npz,
        "n_train": int(X_tr.shape[0]),
        "n_val": int(X_va.shape[0]),
        "n_test": int(X_te.shape[0]),
        "n_features": int(X_tr.shape[1]),
        "class_names": class_names,
        "seed": args.seed,
        "source": {
            "data_dir": args.data_dir,
            "features_file": args.features_file,
            "label_col": args.label_col,
            "id_col": args.id_col,
            "drop_cols": args.drop_cols,
            "top_k": args.top_k,
            "min_per_class": args.min_per_class,
            "limit": args.limit
        }
    }
    meta_path = os.path.join(args.outdir, f"{tag}.meta.json")
    dump_json(meta, meta_path)

    print(f"[OK] Saved dataset NPZ: {out_npz}")
    print(f"[OK] Saved metadata JSON: {meta_path}")
    print(f"[INFO] Shapes  train={X_tr.shape}, val={X_va.shape}, test={X_te.shape}, features={len(feat_names)}")
    print(f"[INFO] Classes {len(class_names)}: {class_names[:10]}{'...' if len(class_names)>10 else ''}")

if __name__ == "__main__":
    main()
