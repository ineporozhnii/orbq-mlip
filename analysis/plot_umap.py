"""UMAP projections of metal atomic embeddings (paper Figs. 3 and 5).

1. Save backbone embeddings of a trained model on the test set::

       orbq-test --ckpt_path runs/<run> --data_path $ORBQ_DATA_DIR/lmdb \
           --split_file $ORBQ_DATA_DIR/splits/datasplits_100000.json --save_emb

   This writes ``runs/<run>/test_<charges_type>_results.pt`` with ``atomic_emb``.

2. Plot one panel per result file::

       python analysis/plot_umap.py --out umap.png \
           --panel "Energy + orbital GFN1-xTB charges" runs/a/test_orbital_dxtb_results.pt \
           --panel "Energy only" runs/b/test_none_results.pt

Requires the ``analysis`` extra (umap-learn); ``adjustText`` is used if installed.
"""

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# transition metals shown in Fig. 3 (override with --elements)
DEFAULT_ELEMENTS = [
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Mo",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Ta",
    "W",
    "Ir",
    "Pt",
    "Au",
]

SYMBOLS = (
    "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb "
    "Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi"
).split()


def plot_panels(panels, elements, out, max_atoms=None, seed=42):
    import umap

    try:
        from adjustText import adjust_text
    except ImportError:
        adjust_text = None

    z_of = {s: i for i, s in enumerate(SYMBOLS)}
    zs = [z_of[e] for e in elements]
    cmap = plt.get_cmap("tab20")
    color = {z: cmap(i % 20) for i, z in enumerate(zs)}

    ncol = min(3, len(panels))
    nrow = math.ceil((len(panels) + 1) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.7 * ncol, 4.5 * nrow), squeeze=False)
    axes = axes.flatten()
    seen = set()

    for ax, (title, path) in zip(axes, panels):
        res = torch.load(path, map_location="cpu", weights_only=False)
        emb = np.asarray(res["atomic_emb"][:max_atoms])
        z = np.asarray(res["atom_types"][:max_atoms])
        mask = np.isin(z, zs)
        emb, z = emb[mask].reshape(mask.sum(), -1), z[mask]
        if len(z) < 3:
            ax.text(0.5, 0.5, "no metal atoms", ha="center", va="center")
            ax.axis("off")
            continue
        xy = umap.UMAP(
            n_neighbors=min(15, len(z) - 1), min_dist=0.1, random_state=seed
        ).fit_transform(emb)
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=[color[v] for v in z],
            s=30,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.3,
        )
        texts = []
        for v in np.unique(z):
            cx, cy = np.median(xy[z == v], axis=0)
            seen.add(v)
            texts.append(
                ax.text(
                    cx,
                    cy,
                    SYMBOLS[v],
                    fontsize=14,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc=(1, 1, 1, 0.8), ec="black", lw=0.5),
                )
            )
        if adjust_text is not None:
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
        ax.set_title(title, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(panels) :]:
        ax.axis("off")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=SYMBOLS[v],
            markerfacecolor=color[v],
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=0.5,
        )
        for v in zs
        if v in seen
    ]
    axes[-1].legend(handles=handles, loc="center", ncol=3, frameon=False, fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--panel",
        nargs=2,
        action="append",
        metavar=("TITLE", "RESULTS_PT"),
        required=True,
        help="Panel title and orbq-test results file saved with --save_emb (repeatable).",
    )
    parser.add_argument(
        "--elements", nargs="+", default=DEFAULT_ELEMENTS, help="Element symbols to include."
    )
    parser.add_argument(
        "--max-atoms", type=int, default=None, help="Use only the first N atoms of each file."
    )
    parser.add_argument("--out", default="umap_metal_clusters.png")
    args = parser.parse_args(argv)
    plot_panels(args.panel, args.elements, args.out, args.max_atoms)


if __name__ == "__main__":
    main()
