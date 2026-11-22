"""``orbq-datagen <step> [options]``: entry point for the data pipeline steps."""

import importlib
import sys

STEPS = {
    "extract": ("orbq.datagen.extract", "select structures from OMol25"),
    "dxtb": ("orbq.datagen.dxtb_charges", "compute GFN1-xTB orbital charges with dxtb"),
    "prepare": ("orbq.datagen.prepare", "filter, split, energy reference, pack"),
    "preprocess": ("orbq.datagen.preprocess", "build the graph LMDB dataset"),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in STEPS:
        print("usage: orbq-datagen <step> [options]\n\nsteps:")
        for name, (_, help_text) in STEPS.items():
            print(f"  {name:<11} {help_text}")
        print("\nRun 'orbq-datagen <step> --help' for the options of a step.")
        sys.exit(0 if argv and argv[0] in ("-h", "--help") else 2)
    module = importlib.import_module(STEPS[argv[0]][0])
    module.main(argv[1:])


if __name__ == "__main__":
    main()
