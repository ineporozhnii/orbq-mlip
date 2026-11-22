"""Data generation pipeline: OMol25 structures -> GFN1-xTB orbital charges -> LMDB graphs.

Steps (run in order, each is also a module with its own ``--help``):

1. ``extract``    - select neutral singlet metal complexes from OMol25
2. ``dxtb``       - GFN1-xTB orbital/atomic charges with dxtb (once per SCF mode)
3. ``prepare``    - cross-mode filtering, splits, energy reference, packing
4. ``preprocess`` - radius graphs into an LMDB dataset
"""
