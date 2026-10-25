#!/usr/bin/env python3
"""Create a reviewed binary crop dataset without modifying the source files.

The index lists refer to sorted filenames in dataset_cls_clean/{split}/other_gesture.
Reviewed from contact sheets on 2026-10-25. Ambiguous crops are excluded.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "dataset_cls_clean"
DEST = ROOT / "dataset_cls_reviewed"
REVIEW = {
    "train": {
        "thumbs_up": {1, 3, 4, 5, 6, 11, 12, 13, 14, 19, 23, 25, 29,
                      31, 32, 33, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46,
                      47, 49, 50, 51, 52, 53, 54, 55, 56, 57, 60, 61, 63},
        "no_thumbs_up": {0, 2, 7, 8, 9, 10, 15, 16, 17, 18, 20, 21, 22, 24,
                          26, 27, 28, 34, 37, 43, 48, 58, 59, 62, 64, 65},
    },
    "val": {
        "thumbs_up": {1, 3, 11, 12},
        "no_thumbs_up": {0, 2, 4, 5, 6, 7, 8, 9, 10, 13, 14},
    },
}


def main() -> None:
    manifest = {"review": {}, "counts": {}}
    for split, labels in REVIEW.items():
        source_negatives = sorted((SOURCE / split / "other_gesture").glob("*.jpg"))
        assert labels["thumbs_up"].isdisjoint(labels["no_thumbs_up"])
        reviewed = labels["thumbs_up"] | labels["no_thumbs_up"]
        assert reviewed <= set(range(len(source_negatives)))
        counts = {"thumbs_up": 0, "no_thumbs_up": 0}
        for folder in counts:
            (DEST / split / folder).mkdir(parents=True, exist_ok=True)
        for path in sorted((SOURCE / split / "thumbs_up").glob("*.jpg")):
            shutil.copy2(path, DEST / split / "thumbs_up" / path.name)
            counts["thumbs_up"] += 1
        for index, path in enumerate(source_negatives):
            if index not in reviewed:
                continue
            folder = "thumbs_up" if index in labels["thumbs_up"] else "no_thumbs_up"
            shutil.copy2(path, DEST / split / folder / path.name)
            counts[folder] += 1
            if split == "train" and folder == "no_thumbs_up":
                for repeat in range(1, 5):
                    shutil.copy2(path, DEST / split / folder / f"{path.stem}_repeat{repeat}.jpg")
                    counts[folder] += 1
        manifest["review"][split] = {name: [source_negatives[i].name for i in sorted(indices)]
                                     for name, indices in labels.items()}
        manifest["review"][split]["excluded_ambiguous"] = [source_negatives[i].name for i in sorted(set(range(len(source_negatives))) - reviewed)]
        manifest["counts"][split] = counts
    (DEST / "review_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
