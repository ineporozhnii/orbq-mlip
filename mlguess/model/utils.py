import os
import torch 
import torch.optim as optim
from bisect import bisect
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import BasePredictionWriter
import matplotlib.pyplot as plt
import numpy as np

def warmup_lr_milestone(current_step: int, optim_config):
    """
    adapted from https://github.com/Open-Catalyst-Project/ocp
    Returns a learning rate multiplier.
    Till `warmup_steps`, learning rate linearly increases to `initial_lr`,
    and then gets multiplied by `lr_gamma` every time a milestone is crossed.
    """

    # keep this block for older configs that have warmup_epochs instead of warmup_steps
    # and lr_milestones are defined in epochs
    if (
        any(x < 100 for x in optim_config["lr_milestones"])
        or "warmup_epochs" in optim_config
    ):
        raise Exception(
            "ConfigError: please define lr_milestones in steps not epochs and define warmup_steps instead of warmup_epochs"
        )

    if current_step <= optim_config["warmup_steps"]:
        alpha = current_step / float(optim_config["warmup_steps"])
        return optim_config["warmup_factor"] * (1.0 - alpha) + alpha
    else:
        idx = bisect(optim_config["lr_milestones"], current_step)
        return pow(optim_config["lr_gamma"], idx)
    
def warm_lr_lambda(current_step: int, 
                   warmup_steps: int = 0, 
                   warmup_factor: float = 0.2,
                   alpha: float = 0.96, 
                   beta: float = 1e4):
    if current_step <= warmup_steps:
        alpha = current_step / float(warmup_steps)
        return warmup_factor * (1.0 - alpha) + alpha
    else:
        return alpha ** (current_step / beta)
            
class PowerDecayScheduler(optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps=1000, warmup_factor=0.2, alpha=0.96, beta=1e4):
        scheduler_fn = lambda step: warm_lr_lambda(step, warmup_steps, warmup_factor, alpha, beta)
        super().__init__(optimizer=optimizer, lr_lambda=scheduler_fn)

class WarmupMilestoneScheduler(optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, config):
        self.config = config
        scheduler_fn = lambda step: warmup_lr_milestone(step, self.config)
        super().__init__(optimizer=optimizer, lr_lambda=scheduler_fn)
    

class PredictionScatterPlotCallback(Callback):
    def __init__(self, log_interval: int = 1,  max_charges_plot: int = 200_000):
        """
        Callback to log scatter plot of predictions vs ground truth to WandB.

        Args:
            log_interval: Interval (in epochs) at which to log the scatter plot.
            max_charges_plot: Maximum number of orbital charges to plot in the scatter plot. (Above 200k WandB interactive plots become slow)
        """
        self.log_interval = log_interval
        self.max_charges_plot = max_charges_plot
        self.n_atom = []
        self.atom_types = []
        self.coords = []
        self.predictions = []
        self.ground_truth = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Collect predictions and ground truth after each validation batch.
        """
        outputs = pl_module(batch)
        preds = outputs[1]
        targets = outputs[2]
        self.n_atom.extend(batch["n_atom"].detach().cpu().numpy())
        self.atom_types.extend(batch["atom_types"].detach().cpu().numpy())
        self.coords.extend(batch["coords"].detach().cpu().numpy())
        self.predictions.extend(preds.detach().cpu().numpy())
        self.ground_truth.extend(targets.detach().cpu().numpy())

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Generate and log the scatter plot at the end of the validation epoch.
        """
        if trainer.current_epoch % self.log_interval == 0:
            # Create scatter plot
            fig = plt.figure(figsize=(8, 8))
        
            plt.scatter(self.ground_truth[:self.max_charges_plot], self.predictions[:self.max_charges_plot], alpha=0.2)
            # Add 1:1 line
            min_val = min(min(self.ground_truth), min(self.predictions))
            max_val = max(max(self.ground_truth), max(self.predictions))
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.5)
            plt.xlabel("Ground Truth")
            plt.ylabel("Predictions")
            plt.title("Predictions vs Ground Truth")
            plt.grid(True)
            trainer.logger.experiment.log({"scatter_plot": fig})
            plt.close('all')
            
            # n_atom = np.array(self.n_atom)
            # atom_types = np.array(self.atom_types)
            # coords = np.array(self.coords)
            # ground_truth = np.array(self.ground_truth)
            # predictions = np.array(self.predictions)
            # save_path = f"{self.save_dir}/orbital_charges_epoch_{trainer.current_epoch}.npz"
            # np.savez(
            #     save_path,
            #     n_atom=n_atom,
            #     atom_types=atom_types,
            #     coords=coords,
            #     ground_truth=ground_truth,
            #     predictions=predictions
            # )

        # Clear stored predictions and ground truth for the next epoch
        self.n_atom = []
        self.atom_types = []
        self.coords = []
        self.predictions = []
        self.ground_truth = []


class ValidationDataLogger(Callback):
    def __init__(self, save_dir="./"):
        super().__init__()
        self.save_dir = save_dir
        self.outputs = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Called when the validation batch ends."""
        if outputs:
            # Move outputs to CPU to save GPU memory
            cpu_outputs = {key: val.cpu() for key, val in outputs.items()}
            self.outputs.append(cpu_outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the validation epoch ends."""
        if not self.outputs or trainer.sanity_checking:
            return

        # 1. Concatenate outputs from all batches on the current GPU
        preds = torch.cat([x['preds'] for x in self.outputs])
        targets = torch.cat([x['targets'] for x in self.outputs])
        # Assuming these tensors have the same first dimension as preds
        n_atom = torch.cat([x['n_atom'] for x in self.outputs])
        atom_types = torch.cat([x['atom_types'] for x in self.outputs])
        coords = torch.cat([x['coords'] for x in self.outputs])

        # Move tensors to the correct device for gathering
        preds = preds.to(pl_module.device)
        targets = targets.to(pl_module.device)
        n_atom = n_atom.to(pl_module.device)
        atom_types = atom_types.to(pl_module.device)
        coords = coords.to(pl_module.device)

        # 2. Pad tensors to the same size
        local_size = torch.tensor(preds.shape[0], device=pl_module.device)
        
        # Gather the sizes of the first dimension of 'preds' from all GPUs
        all_sizes = pl_module.all_gather(local_size)

        max_size = torch.max(all_sizes).item()

        # Pad all tensors to the max_size
        if local_size < max_size:
            pad_size = max_size - local_size
            
            # Create padding with appropriate dimensions
            preds_pad = torch.zeros(pad_size, *preds.shape[1:], dtype=preds.dtype, device=preds.device)
            targets_pad = torch.zeros(pad_size, *targets.shape[1:], dtype=targets.dtype, device=targets.device)
            n_atom_pad = torch.zeros(pad_size, *n_atom.shape[1:], dtype=n_atom.dtype, device=n_atom.device)
            atom_types_pad = torch.zeros(pad_size, *atom_types.shape[1:], dtype=atom_types.dtype, device=atom_types.device)
            coords_pad = torch.zeros(pad_size, *coords.shape[1:], dtype=coords.dtype, device=coords.device)
            
            preds = torch.cat((preds, preds_pad), dim=0)
            targets = torch.cat((targets, targets_pad), dim=0)
            n_atom = torch.cat((n_atom, n_atom_pad), dim=0)
            atom_types = torch.cat((atom_types, atom_types_pad), dim=0)
            coords = torch.cat((coords, coords_pad), dim=0)

        # 3. Gather padded outputs from all GPUs
        # This operation must be called on all processes
        all_preds = pl_module.all_gather(preds)
        all_targets = pl_module.all_gather(targets)
        all_n_atom = pl_module.all_gather(n_atom)
        all_atom_types = pl_module.all_gather(atom_types)
        all_coords = pl_module.all_gather(coords)

        # 4. Save aggregated data on rank 0
        if trainer.global_rank == 0:
            # Reshape and unpad gathered tensors for saving
            # Example: from (num_gpus, max_size, ...) to (total_samples, ...)
            
            preds_list = []
            targets_list = []
            n_atom_list = []
            atom_types_list = []
            coords_list = []

            for i in range(all_sizes.shape[0]):
                size = all_sizes[i].item()
                preds_list.append(all_preds[i][:size])
                targets_list.append(all_targets[i][:size])
                n_atom_list.append(all_n_atom[i][:size])
                atom_types_list.append(all_atom_types[i][:size])
                coords_list.append(all_coords[i][:size])
                
            flat_preds = torch.cat(preds_list).cpu().numpy()
            flat_targets = torch.cat(targets_list).cpu().numpy()
            flat_n_atom = torch.cat(n_atom_list).cpu().numpy()
            flat_atom_types = torch.cat(atom_types_list).cpu().numpy()
            flat_coords = torch.cat(coords_list).cpu().numpy()

            save_path = f"{self.save_dir}/orbital_charges_epoch_{trainer.current_epoch}.npz"
            print(f"Saving validation data to {save_path}")
            np.savez(
                save_path,
                n_atom=flat_n_atom,
                atom_types=flat_atom_types,
                coords=flat_coords,
                targets=flat_targets,
                predictions=flat_preds
            )

        # 5. Clean up
        self.outputs.clear()