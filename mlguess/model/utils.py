import torch 
import torch.optim as optim
from bisect import bisect
from lightning.pytorch.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np

from mlguess.common.utils import scatter

def get_nmape(pred_density, density, batch=None):
    diff = pred_density - density
    sum_over = list(range(1, diff.dim()))
    if batch is None:
        return torch.abs(diff).sum(dim=sum_over) / torch.abs(density).sum(dim=sum_over) * 100.
    else:
        return (scatter(diff.abs(), batch, dim_size=batch.max() + 1).abs() / 
                scatter(density.abs(), batch, dim_size=batch.max() + 1).abs() * 100.)

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
    def __init__(self, log_interval: int = 1, max_charges_plot: int = 200_000):
        """
        Callback to log scatter plot of predictions vs ground truth to WandB.

        Args:
            log_interval: Interval (in epochs) at which to log the scatter plot.
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
            
            n_atom = np.array(self.n_atom)
            atom_types = np.array(self.atom_types)
            coords = np.array(self.coords)
            ground_truth = np.array(self.ground_truth)
            predictions = np.array(self.predictions)
            save_path = f"/home/ihor/projects/MLGuessQC/mlguess/results/tmQMneutral_FPCmatch_filtered_charges_overfit/orbital_charges_epoch_{trainer.current_epoch}.npz"
            np.savez(
                save_path,
                n_atom=n_atom,
                atom_types=atom_types,
                coords=coords,
                ground_truth=ground_truth,
                predictions=predictions
            )

        # Clear stored predictions and ground truth for the next epoch
        self.n_atom = []
        self.atom_types = []
        self.coords = []
        self.predictions = []
        self.ground_truth = []