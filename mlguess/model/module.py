import math
import torch
from lightning import LightningModule
from hydra.utils import instantiate
from torch_ema import ExponentialMovingAverage

from mlguess.common.utils import scatter
from mlguess.model.basis_set import get_basis_set, transform_basis_set, aug_etb_for_basis
from mlguess.model.gtos import GTOs
from mlguess.model.utils import get_nmape

class ChgLightningModule(LightningModule):
    """
    Charge density prediction with the probe point method.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.construct_orbitals()
        num_neighbors = self.hparams.metadata["avg_num_neighbors"]
        
        self.model = instantiate(
            self.hparams.model, 
            num_neighbors=num_neighbors, 
            expo_trainable=self.hparams.expo_trainable,
            max_n_Ls=self.max_n_Ls,
            max_n_orbitals_per_L=self.max_n_orbitals_per_L
        )
        
        self.ema = ExponentialMovingAverage(
            self.parameters(), decay=self.hparams.train.ema.decay
        )        
        self.distributed = ((self.hparams.train.trainer.strategy == "ddp") and 
                            (self.hparams.train.trainer.devices > 1))
        self.register_buffer("scale", torch.FloatTensor([self.hparams.metadata["target_var"]]).sqrt())

    def construct_orbitals(self):
        # construct GTOs
        unique_atom_types = self.hparams.metadata['unique_atom_types']
        basis_set = transform_basis_set(get_basis_set(self.hparams.dft_basis_set))
        if self.hparams.dft_wt_aug:
            basis_set = aug_etb_for_basis(
                basis_set, 
                beta=self.hparams.beta, 
                lmax_restriction=self.hparams.lmax_restriction,
                lmax_relax=self.hparams.lmax_relax if 'lmax_relax' in self.hparams else 0,
            )

        vbasis = basis_set[self.hparams.vnode_elem]
                    
        if self.hparams.uncontracted:
            for v in basis_set.values():
                v['contraction'] = None
            vbasis['contraction'] = None
            
        gto_dict = {}
        for elem in unique_atom_types:
            # atomic number 0 for virtual nodes.
            if elem == 0:
                gto_dict['0'] = GTOs(**vbasis, cutoff=self.hparams.orb_cutoff)
            else:
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
    
    def predict_coeffs(self, batch):
        """
        predict coefficient for GTO basis functions.
        """
        coeffs, expo_scaling = self.model(batch)
        
        #n_orbs = self.n_orbitals[
        #    (batch.atom_types.repeat(len(self.unique_atom_types), 1).T == 
        #     self.unique_atom_types).nonzero()[:, 1]]
        #batch_n_orbs = scatter(n_orbs, batch.batch, len(batch))
        #coeffs = coeffs / batch_n_orbs.repeat_interleave(
        #    batch.n_atom + batch.n_vnode).sqrt().view(-1, 1)

        if expo_scaling is not None:
            # range from 0.5 to 2.0
            expo_scaling = 1.5 / (1 + torch.exp(-expo_scaling + math.log(2))) + 0.5
        
        return coeffs, expo_scaling
    
    def orbital_inference(self, batch, coeffs):
        """
        Compute chg values at given probe points using <coeffs>.
        Inputs:
            - batch: batch (bsz B) object, N atoms
            - coeffs: orbital coefficients (N, max_orbital_outdim)
            - n_probes: number of probes for each batch (B,)
            - probe_coords: probe coordinates (M, 3)
        Outputs:
            - orbitals: chg values at probe points, (M,)
        """

        # Charge constraint (assume 0 charge for all molecules)
        # TODO: extend to non-zero charges
        n_orb_per_mol = scatter(self.orb_index[batch.atom_types].sum(dim=-1), batch.batch, len(batch))
        charges_pad = torch.zeros(self.orb_index[batch.atom_types].shape, device=coeffs.device)
        charges_pad[torch.where(self.orb_index[batch.atom_types] == True)] = coeffs[torch.where(self.orb_index[batch.atom_types] == True)]
        
        mol_charges = scatter(charges_pad.sum(dim=-1), batch.batch, len(batch))
        mol_charge_correction = mol_charges/n_orb_per_mol
        
        _, n_atom_per_mol = torch.unique(batch.batch, return_counts=True)
        corrected_charges = charges_pad - torch.repeat_interleave(mol_charge_correction, n_atom_per_mol).unsqueeze(1)
        
        mask = self.orb_index[batch.atom_types].flatten()
        orbital_charges = corrected_charges.flatten()[mask]
        # Get original predictions 
        #coeffs_masked = coeffs.flatten()[mask]

        pred = orbital_charges * self.scale
    
        return pred
    
    def forward(self, batch):
        coeffs, expo_scaling = self.predict_coeffs(batch)
        pred = self.orbital_inference(batch, coeffs)        
        
        target = batch.y
        if self.hparams.criterion == 'mse':
            loss = (pred / self.scale - target / self.scale).pow(2).mean()
        else:
            loss = (pred / self.scale - target / self.scale).abs().mean()
            
        return loss, pred, batch.y, coeffs, expo_scaling
    
    def training_step(self, batch, batch_idx):        
        loss, pred, target, coeffs, scaling = self(batch)
        
        mse = (pred - target).pow(2).mean().item()
        mae = (pred - target).abs().mean().item()
        self.log_dict({
            "loss/train": loss,
            "mse/train": mse,
            "mae/train": mae,
            }, 
            batch_size=batch["cell"].shape[0], 
            sync_dist=self.distributed
        )   
        
        if scaling is not None:
            self.log_dict({
                "trainer/scaling_mean": scaling.mean(),
                "trainer/scaling_std": scaling.std()
                }, 
                batch_size=batch["cell"].shape[0], 
                sync_dist=self.distributed
            )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, target, _, _ = self(batch)
        
        mse = (pred - target).pow(2).mean().item()
        mae = (pred - target).abs().mean().item()
        self.log_dict({
            "loss/val": loss,
            "mse/val": mse,
            "mae/val": mae,
            }, 
            batch_size=batch["cell"].shape[0],
            sync_dist=self.distributed
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, pred, target, _, _ = self(batch)
        
        mse = (pred - target).pow(2).mean().item()
        mae = (pred - target).abs().mean().item()
        self.log_dict({
            "loss/test": loss,
            "mse/test": mse,
            "mae/test": mae,
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