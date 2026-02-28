import torch
from lightning import LightningModule
from hydra.utils import instantiate
from torch_ema import ExponentialMovingAverage

from mlguess.common.utils import scatter
from mlguess.model.utils import UncertaintyWeightedLoss, UncertaintyWeightedMAELoss
from mlguess.model.basis_set import get_basis_set, transform_basis_set
from mlguess.model.gtos import GTOs

class EnergyChargesLightningModule(LightningModule):
    """
    Energy and orbital/atomic charge prediction module.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.construct_orbitals()
        num_neighbors = self.hparams.metadata["avg_num_neighbors"]
        
        self.model = instantiate(
            self.hparams.model, 
            num_neighbors=num_neighbors, 
            max_n_Ls=self.max_n_Ls,
            max_n_orbitals_per_L=self.max_n_orbitals_per_L
        )

        if self.hparams.train.get("freeze_backbone", False):
            self.freeze_backbone()
        
        if self.hparams.target_property == 'combined':
            self.loss_fn = UncertaintyWeightedLoss() if self.hparams.criterion == 'mse' else UncertaintyWeightedMAELoss()
        
        self.ema = ExponentialMovingAverage(
            self.parameters(), decay=self.hparams.train.ema.decay
        )        
        self.distributed = ((self.hparams.train.trainer.strategy == "ddp") and 
                            (self.hparams.train.trainer.devices > 1))
        self.register_buffer("scale_energy", torch.FloatTensor([self.hparams.metadata["energy_var"]]).sqrt())
        self.register_buffer("scale_atomic_charges_dxtb", torch.FloatTensor([self.hparams.metadata["atomic_charges_dxtb_var"]]).sqrt())
        self.register_buffer("scale_atomic_charges_lowdin", torch.FloatTensor([self.hparams.metadata["atomic_charges_lowdin_var"]]).sqrt())
        self.register_buffer("scale_atomic_charges_mulliken", torch.FloatTensor([self.hparams.metadata["atomic_charges_mulliken_var"]]).sqrt())
        self.register_buffer("scale_orbital_charges_dxtb", torch.FloatTensor([self.hparams.metadata["orbital_charges_dxtb_var"]]).sqrt())

    def construct_orbitals(self):
        unique_atom_types = self.hparams.metadata['unique_atom_types']
        basis_set = transform_basis_set(get_basis_set(self.hparams.dft_basis_set))
            
        gto_dict = {}
        for elem in unique_atom_types:
            gto_dict[str(elem)] = GTOs(**basis_set[elem], cutoff=self.hparams.orb_cutoff)
        
        self.register_buffer('unique_atom_types', torch.tensor(unique_atom_types))
        self.register_buffer('n_Ls', torch.tensor([len(gto_dict[str(i)].Ls) for i in unique_atom_types]))
        self.register_buffer('n_orbitals', torch.tensor([gto_dict[str(i)].outdim for i in unique_atom_types]))
        self.gto_dict = torch.nn.ModuleDict(gto_dict)
        
        self.Lmax = max([gto.Lmax for gto in self.gto_dict.values()])
        self.max_n_Ls = max([len(gto.Ls) for gto in self.gto_dict.values()])
        self.max_n_orbitals_per_L = torch.stack(
            [x.n_orbitals_per_L for x in self.gto_dict.values()]).max(dim=0)[0]
        self.max_outdim_per_L = self.max_n_orbitals_per_L * (2 * torch.arange(len(self.max_n_orbitals_per_L)) + 1)
        self.max_outdim = int(self.max_outdim_per_L.sum())
        
        orb_index = torch.zeros(max(unique_atom_types)+1, self.max_outdim, dtype=torch.bool)
        orb_per_shell = torch.zeros(max(unique_atom_types)+1, 2, dtype=torch.int64)
        offsets = torch.cat([torch.tensor([0]), torch.cumsum(self.max_outdim_per_L, dim=0)])        
        L_index = torch.zeros(max(unique_atom_types)+1, self.max_n_Ls, dtype=torch.bool)
        for k, v in gto_dict.items():
            index = torch.cat(
                [torch.arange(offsets[l], offsets[l]+v.outdim_per_L[l]) for l in range(self.Lmax+1)])
            orb_index[int(k), index] = True
            L_index[int(k), :len(v.Ls)] = True
            orb_per_shell[int(k), 0] = v.orb_per_shell[0]
            orb_per_shell[int(k), 1] = v.orb_per_shell[1]
            
        self.register_buffer('orb_per_shell', orb_per_shell)
        self.register_buffer('orb_index', orb_index)
        self.register_buffer('L_index', L_index)
        self.pbc = self.hparams.pbc
    
    
    def orbital_charge_correction(self, batch, orbital_charges):
        """
        apply correction to the predicted orbital charges to ensure that the total charge of each molecule is correct.
      
        charge constraint (assume total charge == 0 for all molecules)
        TODO: extend to non-zero total charge
        """
        n_orb_per_mol = scatter(self.orb_index[batch.atom_types].sum(dim=-1), batch.batch, len(batch))
        charges_pad = torch.zeros(self.orb_index[batch.atom_types].shape, device=orbital_charges.device)
        charges_pad[torch.where(self.orb_index[batch.atom_types] == True)] = orbital_charges[torch.where(self.orb_index[batch.atom_types] == True)]
        
        mol_charges = scatter(charges_pad.sum(dim=-1), batch.batch, len(batch))
        mol_charge_correction = mol_charges/n_orb_per_mol
        
        _, n_atom_per_mol = torch.unique(batch.batch, return_counts=True)
        corrected_charges = charges_pad - torch.repeat_interleave(mol_charge_correction, n_atom_per_mol).unsqueeze(1)
        mask = self.orb_index[batch.atom_types].flatten()
    
        return corrected_charges.flatten()[mask]
    
    def forward(self, batch):
        energy, orbital_charges, atomic_charges = self.model(batch)
        
        pred_energy = energy * self.scale_energy
        target_energy = batch.energy
        
        if self.hparams.charges_type == 'orbital_dxtb':
            target_charges = batch.orbital_charges_dxtb
            pred_charges = self.orbital_charge_correction(batch, orbital_charges) * self.scale_orbital_charges_dxtb
            scale_charges = self.scale_orbital_charges_dxtb
        elif self.hparams.charges_type == 'atomic_dxtb':
            target_charges = batch.atomic_charges_dxtb
            pred_charges = atomic_charges * self.scale_atomic_charges_dxtb
            scale_charges = self.scale_atomic_charges_dxtb
        elif self.hparams.charges_type == 'atomic_lowdin':
            target_charges = batch.atomic_charges_lowdin
            pred_charges = atomic_charges * self.scale_atomic_charges_lowdin
            scale_charges = self.scale_atomic_charges_lowdin
        elif self.hparams.charges_type == 'atomic_mulliken':
            target_charges = batch.atomic_charges_mulliken
            pred_charges = atomic_charges * self.scale_atomic_charges_mulliken
            scale_charges = self.scale_atomic_charges_mulliken
        elif self.hparams.charges_type == 'none':
            target_charges = torch.zeros_like(atomic_charges)
            pred_charges = torch.zeros_like(atomic_charges)
            scale_charges = 1.0
        
        if self.hparams.target_property == 'combined':
            if self.hparams.criterion == 'mse':
                loss_energy = (pred_energy / self.scale_energy - target_energy / self.scale_energy).pow(2).mean()
                loss_charges = (pred_charges / scale_charges - target_charges / scale_charges).pow(2).mean()
                loss = self.loss_fn(loss_energy, loss_charges)
            else:
                loss_energy = (pred_energy / self.scale_energy - target_energy / self.scale_energy).abs().mean()
                loss_charges = (pred_charges / scale_charges - target_charges / scale_charges).abs().mean()
                loss = self.loss_fn(loss_energy, loss_charges)
        else:
            if self.hparams.target_property == 'energy':
                scale = self.scale_energy
                target = target_energy
                pred = pred_energy
            elif self.hparams.target_property == 'charges':
                scale = scale_charges
                target = target_charges
                pred = pred_charges
            
            if self.hparams.criterion == 'mse':
                loss = (pred / scale - target / scale).pow(2).mean()
            elif self.hparams.criterion == 'mae':
                loss = (pred / scale - target / scale).abs().mean()
            else:
                raise ValueError(f"Unsupported criterion: {self.hparams.criterion}")
            
        return loss, pred_energy, target_energy, pred_charges, target_charges
    
    def training_step(self, batch, batch_idx):        
        loss, pred_energy, target_energy, pred_charges, target_charges = self(batch)
        
        mse_energy = (pred_energy - target_energy).pow(2).mean().item()
        mae_energy = (pred_energy - target_energy).abs().mean().item()
        
        mse_charges = (pred_charges - target_charges).pow(2).mean().item()
        mae_charges = (pred_charges - target_charges).abs().mean().item()
        
        self.log_dict({
            "loss/train": loss,
            "mse/train_energy": mse_energy,
            "mae/train_energy": mae_energy,
            "mse/train_charges": mse_charges,
            "mae/train_charges": mae_charges,
            }, 
            batch_size=batch["cell"].shape[0], 
            sync_dist=self.distributed
        )   
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred_energy, target_energy, pred_charges, target_charges = self(batch)
        
        mse_energy = (pred_energy - target_energy).pow(2).mean().item()
        mae_energy = (pred_energy - target_energy).abs().mean().item()
        
        mse_charges = (pred_charges - target_charges).pow(2).mean().item()
        mae_charges = (pred_charges - target_charges).abs().mean().item()

        self.log_dict({
            "loss/val": loss,
            "mse/val_energy": mse_energy,
            "mae/val_energy": mae_energy,
            "mse/val_charges": mse_charges,
            "mae/val_charges": mae_charges,
            }, 
            batch_size=batch["cell"].shape[0],
            sync_dist=self.distributed
        )
        return {
                    "loss": loss,
                    "pred_charges": pred_charges,
                    "pred_energy": pred_energy,
                    "target_charges": target_charges,
                    "target_energy": target_energy,
                    "n_atom": batch.n_atom,
                    "atom_types": batch.atom_types,
                    "coords": batch.coords
                }

    def test_step(self, batch, batch_idx):
        loss, pred_energy, target_energy, pred_charges, target_charges = self(batch)
        
        mse_energy = (pred_energy - target_energy).pow(2).mean().item()
        mae_energy = (pred_energy - target_energy).abs().mean().item()
        
        mse_charges = (pred_charges - target_charges).pow(2).mean().item()
        mae_charges = (pred_charges - target_charges).abs().mean().item()

        self.log_dict({
            "loss/test": loss,
            "mse/test_energy": mse_energy,
            "mae/test_energy": mae_energy,
            "mse/test_charges": mse_charges,
            "mae/test_charges": mae_charges,
            }, 
            batch_size=batch["cell"].shape[0],
            sync_dist=self.distributed
        )
        return loss

    def configure_optimizers(self):
        opt = instantiate(
            self.hparams.train.optim,
            params=self.parameters(),
            _convert_="partial",
        )
        scheduler = instantiate(self.hparams.train.lr_scheduler, optimizer=opt)
        
        if 'lr_schedule_freq' in self.hparams.train:
            scheduler = {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': self.hparams.train.lr_schedule_freq,
                'monitor': self.hparams.train.monitor.metric
            }
            
        return {"optimizer": opt, "lr_scheduler": scheduler, 'monitor': self.hparams.train.monitor.metric}
    
    def on_fit_start(self):
        self.ema.to(self.device)
        
    def on_save_checkpoint(self, checkpoint):
        with self.ema.average_parameters():
            checkpoint["ema_state_dict"] = self.ema.state_dict()
            
    def on_load_checkpoint(self, checkpoint):
        try:
            if "ema_state_dict" in checkpoint:
                self.ema.load_state_dict(checkpoint["ema_state_dict"])
        except Exception as e:
            print(e)
            print("Failed to load EMA state dict. Please make sure this was intended.")
            
            
    def on_validation_epoch_start(self):
        self.ema.store()
        self.ema.copy_to(self.parameters())
        
    def on_validation_epoch_end(self):
        self.ema.restore()
        if isinstance(self.lr_schedulers(), torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_schedulers().step(self.trainer.callback_metrics[self.hparams.train.monitor.metric])
    
    def on_before_zero_grad(self, optimizer):
        self.ema.update(self.parameters())
        
    def on_after_backward(self):
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), float('inf'), norm_type=2.0)
        self.log('trainer/grad_norm', total_norm)
    
    def freeze_backbone(self):
        print("Freezing backbone (all layers except EnergyBlock)...")
        for param in self.model.parameters():
            param.requires_grad = False
        
        found_energy_block = False
        for name, module in self.model.named_modules():
            if "EnergyBlock" in module.__class__.__name__:
                print(f"Unfreezing module: {name} ({module.__class__.__name__})")
                for param in module.parameters():
                    param.requires_grad = True
                found_energy_block = True
        
        if not found_energy_block:
            print("Warning: EnergyBlock not found! All parameters are frozen.")