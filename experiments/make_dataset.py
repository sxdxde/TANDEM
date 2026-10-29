"""Build the Milestone 7 image set: natural photos, real charts, rendered documents.

- natural: COCO val2017 photos downloaded by id (public, http://images.cocodataset.org).
- chart:   matplotlib charts (bar / line / scatter / pie / hist) with seeded random data.
- doc:     PIL-rendered text pages (paragraphs, a table, a code-like listing).
Images go to data/images/<kind>_<n>.jpg; data/manifest.csv records name, kind, prompt.
COCO downloads are third-party files: they are only opened with PIL, never executed.
"""

from __future__ import annotations

import csv
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "data" / "images"
COCO_IDS = [39769, 285, 632, 724, 776, 785, 802, 872, 885, 1000, 1268, 1296, 1353,
            1425, 1490, 1503, 1532, 1584, 1675, 1761, 1818, 2006, 2149, 2153]
NATURAL_PROMPTS = ["Describe this image in detail.", "What is happening in this picture?",
                   "What objects can you see in this image?"]
LOREM = ("The committee reviewed the quarterly results and noted that revenue grew in three "
         "of five regions while costs remained flat. Several members asked for a more detailed "
         "breakdown of the logistics expenses before the next meeting. The chair proposed that "
         "a working group be formed to examine supplier contracts and report back within sixty "
         "days. The proposal was seconded and carried without objection.").split()


def natural(rows: list) -> None:
    for i, cid in enumerate(COCO_IDS):
        path = IMG_DIR / f"natural_{i:02d}.jpg"
        if not path.exists():
            url = f"http://images.cocodataset.org/val2017/{cid:012d}.jpg"
            try:
                with urllib.request.urlopen(url, timeout=20) as r:
                    path.write_bytes(r.read())
                Image.open(path).convert("RGB").save(path, quality=92)
            except Exception as e:  # noqa: BLE001
                print(f"skip coco {cid}: {e}")
                path.unlink(missing_ok=True)
                continue
        rows.append((path.stem, "natural", NATURAL_PROMPTS[i % 3]))


def charts(rows: list) -> None:
    for i in range(10):
        rng = np.random.default_rng(100 + i)
        fig, ax = plt.subplots(figsize=(10, 7.5), dpi=100)
        kind = ["bar", "line", "scatter", "pie", "hist"][i % 5]
        labels = [f"Q{j + 1}" for j in range(6)]
        if kind == "bar":
            ax.bar(labels, rng.integers(10, 100, 6), color=plt.cm.tab10(np.arange(6)))
        elif kind == "line":
            for k in range(3):
                ax.plot(np.cumsum(rng.normal(0, 1, 40)), label=f"series {k + 1}")
            ax.legend()
        elif kind == "scatter":
            ax.scatter(rng.normal(size=120), rng.normal(size=120), c=rng.random(120), cmap="viridis")
        elif kind == "pie":
            ax.pie(rng.integers(5, 40, 5), labels=[f"item {j}" for j in range(5)], autopct="%1.0f%%")
        else:
            ax.hist(rng.normal(size=500), bins=20, color="steelblue")
        ax.set_title(f"Sample {kind} chart {i}")
        path = IMG_DIR / f"chart_{i:02d}.jpg"
        fig.savefig(path)
        plt.close(fig)
        rows.append((path.stem, "chart", "Describe this chart in detail."))


def docs(rows: list) -> None:
    font = ImageFont.load_default(size=22)
    for i in range(8):
        rng = np.random.default_rng(200 + i)
        img = Image.new("RGB", (1024, 768), "white")
        d = ImageDraw.Draw(img)
        d.text((60, 40), f"Report {i + 1}: meeting minutes and findings", fill="black", font=font)
        y = 100
        if i % 4 == 2:  # table
            for r in range(10):
                for c in range(4):
                    d.rectangle([60 + c * 220, y, 60 + (c + 1) * 220, y + 44], outline="black")
                    d.text((72 + c * 220, y + 10), f"{rng.integers(0, 9999)}", fill="black", font=font)
                y += 44
        else:
            words = list(rng.permutation(LOREM * 6))
            while y < 720:
                n = int(rng.integers(8, 13))
                d.text((60, y), " ".join(words[:n]), fill="black", font=font)
                words = words[n:]
                y += 34
        path = IMG_DIR / f"doc_{i:02d}.jpg"
        img.save(path, quality=92)
        rows.append((path.stem, "doc", ["What does this document say?", "Summarize the text in this image.",
                                        "What are the numbers in the table?", "What is the title?"][i % 4]))


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    rows: list = []
    natural(rows)
    charts(rows)
    docs(rows)
    with open(ROOT / "data" / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "kind", "prompt"])
        w.writerows(rows)
    kinds = {k: sum(1 for r in rows if r[1] == k) for k in ("natural", "chart", "doc")}
    print(f"{len(rows)} images: {kinds}")


if __name__ == "__main__":
    main()
