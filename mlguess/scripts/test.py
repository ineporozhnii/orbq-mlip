import logging
import time
import json
import argparse
import contextlib
import numpy as np
from copy import deepcopy
from pathlib import Path
from tqdm.auto import tqdm

import os
import omegaconf
import torch
from lightning.pytorch import seed_everything

from torch.utils.data import Subset
from mlguess.common.pyg import DataLoader
from mlguess.data.dataset import LmdbDataset
from mlguess.data.datamodule import worker_init_fn
from mlguess.model.utils import get_nmape
from mlguess.model.module import ChgLightningModule

pylogger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

def get_data_probe_chunk(input_data, indices):
    data = deepcopy(input_data)
    data['chg_labels'] = input_data.chg_labels[indices]
    data['probe_coords'] = input_data.probe_coords[indices]
    data['n_probe'] = len(indices)
    data['sampled'] = True
    return data

def get_probe_chunks(n_probes, max_n_probe_per_pass):
    batch_size = len(n_probes)
    n_per_pass = []
    probes_to_process = []
    n_probes = torch.clone(n_probes)
    probe_indices = torch.arange(n_probes.sum(), device=n_probes.device)
    
    n_pass = ((n_probes).sum() / max_n_probe_per_pass).ceil().long()
    pass_start_index = 0
    pass_end_index = 0
    for _ in range(n_pass):
        current_pass = torch.zeros(batch_size, dtype=torch.long, device=n_probes.device)
        current_load = 0
        
        for i in range(batch_size):
            if n_probes[i] == 0:
                continue    
            max_points_for_job = (max_n_probe_per_pass - current_load)
            points_to_process = torch.min(torch.tensor([n_probes[i], max_points_for_job]))
            
            current_load += points_to_process
            current_pass[i] = points_to_process
            pass_end_index += int(points_to_process)
            
            # in-place modification done last
            n_probes[i] -= points_to_process
            
            if current_load >= max_n_probe_per_pass:
                break
        n_per_pass.append(current_pass)
        probes_to_process.append(probe_indices[pass_start_index:pass_end_index])
        pass_start_index = pass_end_index
        pass_end_index = pass_start_index
    
    return n_pass, n_per_pass, probes_to_process
    
def main(
    ckpt_path,
    data_path,
    split_file,
    tag='test',
    max_n_graphs=10000,
    batch_size=4,
    max_n_probe=500000,
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
    model = ChgLightningModule.load_from_checkpoint(checkpoint_path=ckpt).to('cuda')
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
        predictions = []
        ground_truth = []
        for batch in test_loader:
            batch = batch.to('cuda')
            coeffs, expo_scaling = model.predict_coeffs(batch)
            pred = model.orbital_inference(batch, coeffs)  
            n_atom.extend(batch["n_atom"].detach().cpu().numpy())
            atom_types.extend(batch["atom_types"].detach().cpu().numpy())
            coords.extend(batch["coords"].detach().cpu().numpy())
            predictions.extend(pred.detach().cpu().numpy())
            ground_truth.extend(batch.y.detach().cpu())
            metadata_list.extend(batch.metadata)
            prog.update(batch["n_atom"].shape[0])        
        elapsed_time = time.time() - curr_time
        pylogger.info(f"elapsed time: {elapsed_time:.2f} seconds.")
        
        n_atom = np.array(n_atom)
        atom_types = np.array(atom_types)
        coords = np.array(coords)
        ground_truth = np.array(ground_truth)
        predictions = np.array(predictions)
        save_path = Path(data_path) / f"{tag}_out.npz"
        np.savez(
            save_path,
            n_atom=n_atom,
            atom_types=atom_types,
            coords=coords,
            ground_truth=ground_truth,
            predictions=predictions,
            metadata=metadata_list,
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
        "--max_n_probe",
        type=int,
        default=180000,
        help="max number of probes to process in one pass.",
    )
    parser.add_argument(
        "--use_last",
        action="store_true",
        help="Use the last checkpoint.",
    )
    args: argparse.Namespace = parser.parse_args()
    main(args.ckpt_path, args.data_path, args.split_file, args.tag, 
         args.max_n_graphs, args.batch_size, args.max_n_probe, args.use_last)