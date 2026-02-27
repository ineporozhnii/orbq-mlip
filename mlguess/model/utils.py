import torch 
import torch.nn as nn
import torch.optim as optim
from bisect import bisect
from lightning.pytorch.callbacks import Callback
import matplotlib.pyplot as plt

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
        
        # Buffers for charges and energy
        self.pred_charges = []
        self.target_charges = []
        self.pred_energy = []
        self.target_energy = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Collect predictions and ground truth after each validation batch.
        """
        # Assume outputs contain the keys described in the prompt
        self.n_atom.extend(batch["n_atom"].detach().cpu().numpy())
        self.atom_types.extend(batch["atom_types"].detach().cpu().numpy())
        self.coords.extend(batch["coords"].detach().cpu().numpy())
        
        if "pred_charges" in outputs:
            self.pred_charges.extend(outputs["pred_charges"].detach().cpu().numpy())
            self.target_charges.extend(outputs["target_charges"].detach().cpu().numpy())
        
        if "pred_energy" in outputs:
            self.pred_energy.extend(outputs["pred_energy"].detach().cpu().numpy())
            self.target_energy.extend(outputs["target_energy"].detach().cpu().numpy())

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Generate and log the scatter plot at the end of the validation epoch.
        """
        if trainer.current_epoch % self.log_interval == 0:
            # Create subplots: one for Charges, one for Energy
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
            # --- Plot Charges ---
            ax_c = axes[0]
            num_plot = min(len(self.target_charges), self.max_charges_plot)
            ax_c.scatter(self.target_charges[:num_plot], self.pred_charges[:num_plot], alpha=0.2)
            
            if len(self.target_charges) > 0:
                min_val = min(min(self.target_charges), min(self.pred_charges))
                max_val = max(max(self.target_charges), max(self.pred_charges))
                ax_c.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.5)
                
            ax_c.set_xlabel("Target Charges")
            ax_c.set_ylabel("Predicted Charges")
            ax_c.set_title("Charges: Preds vs GT")
            ax_c.grid(True)

            # --- Plot Energy ---
            ax_e = axes[1]
            ax_e.scatter(self.target_energy, self.pred_energy, alpha=0.5, c='orange')
            
            if len(self.target_energy) > 0:
                e_min = min(min(self.target_energy), min(self.pred_energy))
                e_max = max(max(self.target_energy), max(self.pred_energy))
                ax_e.plot([e_min, e_max], [e_min, e_max], 'k--', lw=2, alpha=0.5)
            
            ax_e.set_xlabel("Target Energy")
            ax_e.set_ylabel("Predicted Energy")
            ax_e.set_title("Energy: Preds vs GT")
            ax_e.grid(True)

            trainer.logger.experiment.log({"scatter_plot": fig})
            plt.close('all')

        # Clear stored predictions and ground truth for the next epoch
        self.n_atom = []
        self.atom_types = []
        self.coords = []
        self.pred_charges = []
        self.target_charges = []
        self.pred_energy = []
        self.target_energy = []


class ValidationDataLogger(Callback):
    def __init__(self, save_dir="./"):
        super().__init__()
        self.save_dir = save_dir
        self.outputs = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Called when the validation batch ends."""
        if outputs:
            cpu_outputs = {key: val.cpu() for key, val in outputs.items()}
            cpu_outputs['n_atom'] = batch['n_atom'].cpu()
            cpu_outputs['atom_types'] = batch['atom_types'].cpu()
            cpu_outputs['coords'] = batch['coords'].cpu()
            
            self.outputs.append(cpu_outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the validation epoch ends."""
        if not self.outputs or trainer.sanity_checking:
            return

        # List of keys we want to gather and save
        keys_to_gather = [
            'pred_charges', 'target_charges', 
            'pred_energy', 'target_energy', 
            'n_atom', 'atom_types', 'coords'
        ]
        
        final_data = {}
        # Process each key independently to handle different dimensions (e.g. atom-wise vs graph-wise)
        for key in keys_to_gather:
            # Check if key exists in the first batch output
            if key not in self.outputs[0]:
                continue
            
            tensors = [x[key] for x in self.outputs if key in x]
            if not tensors:
                continue
                
            local_tensor = torch.cat(tensors).to(pl_module.device)

            local_size = torch.tensor(local_tensor.shape[0], device=pl_module.device)
            all_sizes = pl_module.all_gather(local_size)
            max_size = torch.max(all_sizes).item()

            if local_size < max_size:
                pad_size = max_size - local_size
                padding = torch.zeros(pad_size, *local_tensor.shape[1:], dtype=local_tensor.dtype, device=local_tensor.device)
                local_tensor = torch.cat((local_tensor, padding), dim=0)

            gathered_tensors = pl_module.all_gather(local_tensor)
            if trainer.global_rank == 0:
                data_list = []
                for i in range(all_sizes.shape[0]):
                    size = all_sizes[i].item()
                    data_list.append(gathered_tensors[i][:size])
                
                final_data[key] = torch.cat(data_list).cpu()

        if trainer.global_rank == 0:
            save_path = f"{self.save_dir}/orbital_charges_epoch_{trainer.current_epoch}.pt"
            print(f"Saving validation data to {save_path}")
            torch.save(final_data, save_path)
            
        self.outputs.clear()
        

class UncertaintyWeightedLoss(nn.Module):
    def __init__(self):
        super(UncertaintyWeightedLoss, self).__init__()
        self.log_var_e = nn.Parameter(torch.tensor(0.0))
        self.log_var_q = nn.Parameter(torch.tensor(0.0))

    def forward(self, loss_e, loss_q):
        """
        Args:
            loss_e (torch.Tensor): The raw scalar loss for energy.
            loss_q (torch.Tensor): The raw scalar loss for charges.
        Returns:
            torch.Tensor: The combined, weighted scalar loss.
        """
        precision_e = torch.exp(-self.log_var_e)
        precision_q = torch.exp(-self.log_var_q)
        
        weighted_loss_e = 0.5 * precision_e * loss_e + 0.5 * self.log_var_e
        weighted_loss_q = 0.5 * precision_q * loss_q + 0.5 * self.log_var_q

        return weighted_loss_e + weighted_loss_q
    

class UncertaintyWeightedMAELoss(nn.Module):
    def __init__(self):
        super(UncertaintyWeightedMAELoss, self).__init__()
        self.log_scale_e = nn.Parameter(torch.zeros(1))
        self.log_scale_q = nn.Parameter(torch.zeros(1))

    def forward(self, loss_e, loss_q):
        total_loss = 0.0

        precision_e = torch.exp(-self.log_scale_e)
        precision_q = torch.exp(-self.log_scale_q)
        
        total_loss = total_loss + (precision_e * loss_e + self.log_scale_e)
        total_loss = total_loss + (precision_q * loss_q + self.log_scale_q)

        return total_loss