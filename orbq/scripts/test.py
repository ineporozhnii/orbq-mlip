import logging
import time
import json
import torch
import argparse
import contextlib
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from lightning.pytorch import seed_everything

from torch.utils.data import Subset
from orbq.common.compat import legacy_pickle_module, upgrade_legacy_targets
from orbq.common.pyg import DataLoader
from orbq.data.dataset import LmdbDataset
from orbq.data.datamodule import worker_init_fn
from orbq.model.module import MultitaskLightningModule

pylogger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


def load_model_from_checkpoint(ckpt_path: Path):
    """Load a trusted local Lightning checkpoint that contains OmegaConf metadata.

    Lightning 2.5+ / PyTorch 2.6 defaults to weights_only=True, which breaks on
    these older checkpoints. We explicitly deserialize the checkpoint with
    weights_only=False and then restore the model + EMA state dict manually.
    """
    checkpoint = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
        pickle_module=legacy_pickle_module,
    )
    hparams = checkpoint.get("hyper_parameters")
    if hparams is None:
        raise KeyError(f"Checkpoint <{ckpt_path}> has no 'hyper_parameters'.")
    # checkpoints trained before the mlguess -> orbq rename
    hparams = upgrade_legacy_targets(hparams)

    model = MultitaskLightningModule(**hparams)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if "ema_state_dict" in checkpoint:
        model.ema.load_state_dict(checkpoint["ema_state_dict"])
    return model


def main(
    ckpt_path,
    data_path,
    split_file,
    tag="test",
    max_n_graphs=10000,
    batch_size=4,
    use_last=False,
    run_name=None,
    save_emb=False,
    device=None,
    num_workers=8,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(42)
    ckpt_path = Path(ckpt_path)

    log_name = f"eval_{tag}.log" if run_name is None else f"eval_{tag}_{run_name}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        filename=ckpt_path / log_name,
        filemode="w",
    )

    # set up data loader
    dataset = LmdbDataset(data_path)
    with open(split_file, "r") as fp:
        splits = json.load(fp)
    test_dataset = Subset(dataset, splits[tag])
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )

    # load checkpoint
    storage_dir: str = Path(ckpt_path)
    if (storage_dir / "last.ckpt").exists() and use_last:
        ckpt = storage_dir / "last.ckpt"
    else:
        ckpts = list(storage_dir.glob("*epoch*.ckpt"))
        if len(ckpts) > 0:
            ckpt_epochs = np.array(
                [int(ckpt.parts[-1].split("-")[0].split("=")[1]) for ckpt in ckpts]
            )
            ckpt_ix = ckpt_epochs.argsort()[-1]
            ckpt = str(ckpts[ckpt_ix])
        else:
            raise FileNotFoundError(f"No checkpoint found in <{ckpt_path}>.")

    # pylint: disable=E1120
    pylogger.info(f"loading checkpoint: {ckpt}")
    model = load_model_from_checkpoint(Path(ckpt)).to(device)
    model.eval()
    model.ema.copy_to(model.parameters())

    # Better load balancing possible.
    pylogger.info("starting testing.")
    with (
        torch.no_grad(),
        contextlib.ExitStack() as context_stack,
    ):

        prog = context_stack.enter_context(
            tqdm(total=min(max_n_graphs, len(test_dataset)), disable=None)
        )
        display_bar = context_stack.enter_context(
            tqdm(
                bar_format=(
                    ""
                    if prog.disable  # prog.ncols doesn't exist if disabled
                    else ("{desc:." + str(prog.ncols) + "}")
                ),
                disable=None,
            )
        )

        curr_time = time.time()

        metadata_list = []
        n_atom = []
        atom_types = []
        coords = []
        predictions_energy = []
        predictions_charges = []
        predictions_forces = []
        ground_truth_energy = []
        ground_truth_charges = []
        ground_truth_forces = []
        atomic_embeddings = []
        for batch in test_loader:
            batch = batch.to(device)
            
            loss, pred_energy, target_energy, pred_charges, target_charges, pred_forces, target_forces = model(batch)

            n_atom.extend(batch["n_atom"].detach().cpu())
            atom_types.extend(batch["atom_types"].detach().cpu())
            coords.extend(batch["coords"].detach().cpu())
            
            if pred_energy is not None:
                predictions_energy.extend(pred_energy.detach().cpu())
            if target_energy is not None:
                ground_truth_energy.extend(target_energy.detach().cpu())
                
            if pred_charges is not None:
                predictions_charges.extend(pred_charges.detach().cpu())
            if target_charges is not None:
                ground_truth_charges.extend(target_charges.detach().cpu())
                
            if pred_forces is not None:
                predictions_forces.extend(pred_forces.detach().cpu().flatten())
            if target_forces is not None:
                ground_truth_forces.extend(target_forces.detach().cpu().flatten())
                
            if save_emb:
                out = model.model(batch)
                if len(out) == 5:
                    _, _, _, _, atomic_emb = out
                else:
                    _, _, _, atomic_emb = out
                atomic_embeddings.extend(atomic_emb.detach().cpu())
                
            metadata_list.extend(batch.metadata)
            prog.update(batch["n_atom"].shape[0])
        elapsed_time = time.time() - curr_time
        pylogger.info(f"elapsed time: {elapsed_time:.2f} seconds.")

        n_atom = torch.tensor(n_atom)
        atom_types = torch.tensor(atom_types)
        predictions_energy = torch.tensor(predictions_energy)
        ground_truth_energy = torch.tensor(ground_truth_energy)
        coords = torch.cat(coords)
        predictions_charges = torch.tensor(predictions_charges)
        ground_truth_charges = torch.tensor(ground_truth_charges)

        base_name = f"{tag}_{model.hparams.charges_type}_results"
        if run_name is not None:
            base_name = f"{base_name}_{run_name}"
        save_path = storage_dir / f"{base_name}.pt"

        save_dict = {
            "n_atom": n_atom,
            "atom_types": atom_types,
            "coords": coords,
            "predictions_energy": predictions_energy,
            "predictions_charges": predictions_charges,
            "ground_truth_energy": ground_truth_energy,
            "ground_truth_charges": ground_truth_charges,
            "metadata": metadata_list,
        }
        if predictions_forces:
            save_dict["predictions_forces"] = torch.tensor(predictions_forces)
        if ground_truth_forces:
            save_dict["ground_truth_forces"] = torch.tensor(ground_truth_forces)
        if save_emb:
            save_dict["atomic_emb"] = (
                torch.stack(atomic_embeddings)
                if atomic_embeddings
                else torch.tensor([])
            )

        torch.save(save_dict, save_path)


def cli():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained orbq checkpoint on a dataset split and save predictions."
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to the checkpoint folder.",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--split_file",
        type=Path,
        help="Path to the split file.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="test",
        help="split to use",
    )
    parser.add_argument(
        "--max_n_graphs",
        type=int,
        default=100_000,
        help="max number of data points to do inference for.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=24,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--use_last",
        action="store_true",
        help="Use the last checkpoint.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Test run name.",
    )
    parser.add_argument(
        "--save_emb",
        action="store_true",
        help="Save atomic embeddings.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of dataloader workers.",
    )
    args: argparse.Namespace = parser.parse_args()
    main(
        args.ckpt_path,
        args.data_path,
        args.split_file,
        args.tag,
        args.max_n_graphs,
        args.batch_size,
        args.use_last,
        args.name,
        args.save_emb,
        args.device,
        args.num_workers,
    )


if __name__ == "__main__":
    cli()
