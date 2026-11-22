# OMol25 sample (23 structures)

A tiny dataset for tests and smoke runs (`orbq-train data=sample train=smoke`):
23 neutral singlet metal complexes from the OMol25 training split, processed with
the `orbq-datagen` pipeline (GFN1-xTB orbital charges via dxtb, DFT energies,
forces, Mulliken and Löwdin charges; energies are reference-corrected).

- `lmdb/data.0000.lmdb` — graph dataset (`orbq.data.dataset.LmdbDataset`)
- `splits/datasplits_15.json` — 15 train / 4 validation / 4 test
- `metadata.json` — normalization statistics

The structures and DFT labels are derived from the Open Molecules 2025 (OMol25)
dataset by Meta FAIR, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):
D. S. Levine et al., *The Open Molecules 2025 (OMol25) Dataset, Evaluations, and
Models*, arXiv:2505.08762. Changes: subset selection, added GFN1-xTB labels,
energy reference correction, conversion to graphs.
