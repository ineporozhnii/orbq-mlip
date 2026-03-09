import logging
import time
import json
import torch
import argparse
import contextlib
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import omegaconf
from lightning.pytorch import seed_everything

from torch.utils.data import Subset
from mlguess.common.pyg import DataLoader
from mlguess.data.dataset import LmdbDataset
from mlguess.data.datamodule import worker_init_fn
from mlguess.model.module import EnergyChargesLightningModule

pylogger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
def main(
    ckpt_path,
    data_path,
    split_file,
    tag='test',
    max_n_graphs=10000,
    batch_size=4,
    use_last=False,
    ):
    seed_everything(42)
    ckpt_path = Path(ckpt_path)
    
    # Configure logging
    logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%m/%d/%Y %I:%M:%S %p',
                filename=ckpt_path / f'eval_{tag}.log',
                filemode='w')
        
    cfg = omegaconf.OmegaConf.load(ckpt_path / "config.yaml")
    
    # set up data loader
    dataset = LmdbDataset(data_path)
    with open(split_file, "r") as fp:
        splits = json.load(fp)
    test_dataset = Subset(dataset, splits[tag])
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=8, 
        worker_init_fn=worker_init_fn
    )

    # load checkpoint
    storage_dir: str = Path(ckpt_path)
    if (storage_dir / 'last.ckpt').exists() and use_last:
        ckpt = storage_dir / 'last.ckpt'
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
    print(f"Loading checkpoint from {ckpt}")
    pylogger.info(f"loaded checkpoint: {ckpt}")
    model = EnergyChargesLightningModule.load_from_checkpoint(checkpoint_path=ckpt).to('cuda')
    model.eval()
    model.ema.copy_to(model.parameters())
    
    # Better load balancing possible.
    pylogger.info("starting testing.")    
    with(torch.no_grad(),
         contextlib.ExitStack() as context_stack, 
         open(storage_dir / f'nmape_{tag}.txt', 'w') as f
        ):
                
        prog = context_stack.enter_context(tqdm(total=min(max_n_graphs, len(test_dataset)), disable=None))
        display_bar = context_stack.enter_context(
                tqdm(
                    bar_format=""
                    if prog.disable  # prog.ncols doesn't exist if disabled
                    else ("{desc:." + str(prog.ncols) + "}"),
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
        ground_truth_energy = []
        ground_truth_charges = []
        for batch in test_loader:
            batch = batch.to('cuda')
            energy, orbital_charges, atomic_charges = model.model(batch)
            
            if model.hparams.charges_type == 'orbital_dxtb':
                target_charges = batch.orbital_charges_dxtb
                pred_charges = model.orbital_charge_correction(batch, orbital_charges) 
            elif model.hparams.charges_type == 'atomic_dxtb':
                target_charges = batch.atomic_charges_dxtb
                pred_charges = atomic_charges   
            elif model.hparams.charges_type == 'atomic_lowdin':
                target_charges = batch.atomic_charges_lowdin
                pred_charges = atomic_charges
            elif model.hparams.charges_type == 'atomic_mulliken':
                target_charges = batch.atomic_charges_mulliken
                pred_charges = atomic_charges
            else:
                target_charges = torch.zeros_like(batch.atom_types)
                pred_charges = torch.zeros_like(batch.atom_types)
                
            n_atom.extend(batch["n_atom"].detach().cpu())
            atom_types.extend(batch["atom_types"].detach().cpu())
            coords.extend(batch["coords"].detach().cpu())
            predictions_energy.extend((energy).detach().cpu())
            predictions_charges.extend((pred_charges).detach().cpu())
            ground_truth_energy.extend(batch.energy.detach().cpu())
            ground_truth_charges.extend(target_charges.detach().cpu())
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
        save_path = storage_dir / f"{tag}_{model.hparams.charges_type}_results.pt"
        torch.save(
            {
                "n_atom": n_atom,
                "atom_types": atom_types,
                "coords": coords,
                "predictions_energy": predictions_energy,
                "predictions_charges": predictions_charges,
                "ground_truth_energy": ground_truth_energy,
                "ground_truth_charges": ground_truth_charges,
                "metadata": metadata_list,
            },
            save_path
        )
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    args: argparse.Namespace = parser.parse_args()
    main(args.ckpt_path, args.data_path, args.split_file, args.tag, 
         args.max_n_graphs, args.batch_size, args.use_last)