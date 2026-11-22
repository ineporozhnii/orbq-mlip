import logging

import torch
from hydra.utils import instantiate
from lightning import LightningModule
from torch_ema import ExponentialMovingAverage

from orbq.common.utils import scatter
from orbq.model.basis_set import get_basis_set, transform_basis_set
from orbq.model.gtos import GTOs

log = logging.getLogger(__name__)


class MultitaskLightningModule(LightningModule):
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
            max_n_orbitals_per_L=self.max_n_orbitals_per_L,
        )

        if self.hparams.train.get("freeze_backbone", False):
            self.freeze_backbone()

        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.hparams.train.ema.decay)
        self.distributed = (
            isinstance(self.hparams.train.trainer.strategy, str)
            and "ddp" in self.hparams.train.trainer.strategy.lower()
        ) and (self.hparams.train.trainer.devices > 1)
        self.register_buffer(
            "scale_energy",
            torch.FloatTensor([self.hparams.metadata["energy_var"]]).sqrt(),
        )
        self.register_buffer(
            "scale_atomic_charges_dxtb",
            torch.FloatTensor([self.hparams.metadata["atomic_charges_dxtb_var"]]).sqrt(),
        )
        self.register_buffer(
            "scale_atomic_charges_lowdin",
            torch.FloatTensor([self.hparams.metadata["atomic_charges_lowdin_var"]]).sqrt(),
        )
        self.register_buffer(
            "scale_atomic_charges_mulliken",
            torch.FloatTensor([self.hparams.metadata["atomic_charges_mulliken_var"]]).sqrt(),
        )
        self.register_buffer(
            "scale_orbital_charges_dxtb",
            torch.FloatTensor([self.hparams.metadata["orbital_charges_dxtb_var"]]).sqrt(),
        )
        self.register_buffer(
            "scale_forces",
            torch.FloatTensor([self.hparams.metadata.get("forces_var", 1.0)]).sqrt(),
        )

    def construct_orbitals(self):
        unique_atom_types = self.hparams.metadata["unique_atom_types"]
        basis_set = transform_basis_set(get_basis_set(self.hparams.dft_basis_set))

        gto_dict = {}
        for elem in unique_atom_types:
            gto_dict[str(elem)] = GTOs(**basis_set[elem], cutoff=self.hparams.orb_cutoff)

        self.register_buffer("unique_atom_types", torch.tensor(unique_atom_types))
        self.register_buffer(
            "n_Ls", torch.tensor([len(gto_dict[str(i)].Ls) for i in unique_atom_types])
        )
        self.register_buffer(
            "n_orbitals",
            torch.tensor([gto_dict[str(i)].outdim for i in unique_atom_types]),
        )
        self.gto_dict = torch.nn.ModuleDict(gto_dict)

        self.Lmax = max([gto.Lmax for gto in self.gto_dict.values()])
        self.max_n_Ls = max([len(gto.Ls) for gto in self.gto_dict.values()])
        self.max_n_orbitals_per_L = torch.stack(
            [x.n_orbitals_per_L for x in self.gto_dict.values()]
        ).max(dim=0)[0]
        self.max_outdim_per_L = self.max_n_orbitals_per_L * (
            2 * torch.arange(len(self.max_n_orbitals_per_L)) + 1
        )
        self.max_outdim = int(self.max_outdim_per_L.sum())

        self.max_matrix_dim_per_L = self.max_n_orbitals_per_L * (
            (2 * torch.arange(len(self.max_n_orbitals_per_L)) + 1) ** 2
        )
        self.max_matrix_dim = int(self.max_matrix_dim_per_L.sum())

        self.max_dataset_matrix_dim_per_L = (
            self.max_n_orbitals_per_L * (2 * torch.arange(len(self.max_n_orbitals_per_L)) + 1)
        ) ** 2
        self.max_dataset_matrix_dim = int(self.max_dataset_matrix_dim_per_L.sum())

        dataset_orb_index = torch.zeros(
            max(unique_atom_types) + 1, self.max_dataset_matrix_dim, dtype=torch.bool
        )
        dataset_to_model_mask = torch.zeros(
            max(unique_atom_types) + 1, self.max_dataset_matrix_dim, dtype=torch.bool
        )

        orb_index = torch.zeros(max(unique_atom_types) + 1, self.max_outdim, dtype=torch.bool)
        matrix_orb_index = torch.zeros(
            max(unique_atom_types) + 1, self.max_matrix_dim, dtype=torch.bool
        )
        diag_in_matrix_index = torch.zeros(
            max(unique_atom_types) + 1, self.max_outdim, dtype=torch.int64
        )

        orb_per_shell = torch.zeros(max(unique_atom_types) + 1, 2, dtype=torch.int64)
        offsets = torch.cat([torch.tensor([0]), torch.cumsum(self.max_outdim_per_L, dim=0)])
        matrix_offsets = torch.cat(
            [torch.tensor([0]), torch.cumsum(self.max_matrix_dim_per_L, dim=0)]
        )
        dataset_matrix_offsets = torch.cat(
            [torch.tensor([0]), torch.cumsum(self.max_dataset_matrix_dim_per_L, dim=0)]
        )
        L_index = torch.zeros(max(unique_atom_types) + 1, self.max_n_Ls, dtype=torch.bool)
        for k, v in gto_dict.items():
            index = torch.cat(
                [
                    torch.arange(offsets[l], offsets[l] + v.outdim_per_L[l])
                    for l in range(self.Lmax + 1)
                ]
            )
            orb_index[int(k), index] = True

            matrix_index_list = []
            diag_mapping_list = []
            for l in range(self.Lmax + 1):
                n_orbs = v.n_orbitals_per_L[l]
                if n_orbs > 0:
                    mat_dim = n_orbs * (2 * l + 1) ** 2
                    matrix_index_list.append(
                        torch.arange(matrix_offsets[l], matrix_offsets[l] + mat_dim)
                    )

                    diag_offsets = torch.arange(2 * l + 1) * (2 * l + 2)
                    for orb_idx in range(n_orbs):
                        diag_mapping_list.append(
                            matrix_offsets[l] + orb_idx * (2 * l + 1) ** 2 + diag_offsets
                        )

                    dim_l = n_orbs * (2 * l + 1)
                    full_mat_dim = dim_l**2
                    block_start = dataset_matrix_offsets[l]
                    dataset_orb_index[int(k), block_start : block_start + full_mat_dim] = True

                    local_mask = torch.zeros(dim_l, dim_l, dtype=torch.bool)
                    for i in range(n_orbs):
                        start = i * (2 * l + 1)
                        end = (i + 1) * (2 * l + 1)
                        local_mask[start:end, start:end] = True

                    dataset_to_model_mask[int(k), block_start : block_start + full_mat_dim] = (
                        local_mask.flatten()
                    )

            if matrix_index_list:
                matrix_index = torch.cat(matrix_index_list)
                matrix_orb_index[int(k), matrix_index] = True

            if diag_mapping_list:
                diag_mapping = torch.cat(diag_mapping_list)
                diag_in_matrix_index[int(k), index] = diag_mapping

            L_index[int(k), : len(v.Ls)] = True
            orb_per_shell[int(k), 0] = v.orb_per_shell[0]
            orb_per_shell[int(k), 1] = v.orb_per_shell[1]

        self.register_buffer("orb_per_shell", orb_per_shell)
        self.register_buffer("orb_index", orb_index)
        self.register_buffer("matrix_orb_index", matrix_orb_index)
        self.register_buffer("dataset_orb_index", dataset_orb_index)
        self.register_buffer("dataset_to_model_mask", dataset_to_model_mask)
        self.register_buffer("diag_in_matrix_index", diag_in_matrix_index)
        self.register_buffer("L_index", L_index)
        self.pbc = self.hparams.pbc

    def orbital_charge_correction(self, batch, orbital_charges_diag):
        """
        apply correction to the predicted orbital charges to ensure that the total charge of each molecule is correct.

        charge constraint (assume total charge == 0 for all molecules)
        """
        n_orb_per_mol = scatter(
            self.orb_index[batch.atom_types].sum(dim=-1), batch.batch, len(batch)
        )
        charges_pad = torch.zeros(
            self.orb_index[batch.atom_types].shape, device=orbital_charges_diag.device
        )
        charges_pad[self.orb_index[batch.atom_types]] = orbital_charges_diag

        mol_charges = scatter(charges_pad.sum(dim=-1), batch.batch, len(batch))
        mol_charge_correction = mol_charges / n_orb_per_mol

        _, n_atom_per_mol = torch.unique(batch.batch, return_counts=True)
        corrected_charges = charges_pad - torch.repeat_interleave(
            mol_charge_correction, n_atom_per_mol
        ).unsqueeze(1)
        mask = self.orb_index[batch.atom_types].flatten()

        return corrected_charges.flatten()[mask]

    def forward(self, batch):
        out = self.model(batch)
        if len(out) == 5:
            energy, forces, orbital_charges, atomic_charges, _ = out
            pred_forces = forces * self.scale_forces if forces is not None else None
        else:
            energy, orbital_charges, atomic_charges, _ = out
            pred_forces = None

        pred_energy = energy * self.scale_energy if energy is not None else None
        target_energy = getattr(batch, "energy", None)

        loss_coeffs = getattr(self.hparams, "loss_coeffs", {})
        loss = 0.0

        pred_charges_diag = None
        target_charges_diag = None
        pred_charges = None
        target_charges = None
        scale_charges = 1.0

        if getattr(self.model, "regress_charges", True) and (
            orbital_charges is not None or atomic_charges is not None
        ):
            if getattr(self.hparams, "charges_type", "none") == "orbital_dxtb":
                full_target_charges = batch.orbital_charges_dxtb
                dset_mask = self.dataset_to_model_mask[batch.atom_types][
                    self.dataset_orb_index[batch.atom_types]
                ]
                target_charges = full_target_charges[dset_mask]

                # Since target_charges are now full matrices, we mask pred_charges with matrix_orb_index
                mask = self.matrix_orb_index[batch.atom_types].flatten()
                pred_charges = orbital_charges.flatten()[mask] * self.scale_orbital_charges_dxtb
                scale_charges = self.scale_orbital_charges_dxtb

                # Extract diagonals for metrics
                charges_mat_pad = torch.zeros(
                    self.matrix_orb_index[batch.atom_types].shape,
                    device=orbital_charges.device,
                )
                charges_mat_pad[self.matrix_orb_index[batch.atom_types]] = pred_charges
                pred_charges_diag_pad = torch.gather(
                    charges_mat_pad, 1, self.diag_in_matrix_index[batch.atom_types]
                )
                pred_charges_diag = pred_charges_diag_pad[self.orb_index[batch.atom_types]]

                # Apply orbital charge correction to enforce total charge constraint
                corrected_charges_diag = self.orbital_charge_correction(batch, pred_charges_diag)

                # Calculate the difference to shift the diagonal
                correction_diff = corrected_charges_diag - pred_charges_diag
                correction_diff_pad = torch.zeros(
                    self.orb_index[batch.atom_types].shape, device=orbital_charges.device
                )
                correction_diff_pad[self.orb_index[batch.atom_types]] = correction_diff

                # Add the difference directly back to the full matrix diagonal
                charges_mat_pad = charges_mat_pad.scatter_add(
                    1, self.diag_in_matrix_index[batch.atom_types], correction_diff_pad
                )

                # Re-extract the corrected full matrix back to pred_charges
                pred_charges = charges_mat_pad[self.matrix_orb_index[batch.atom_types]]

                target_mat_pad = torch.zeros(
                    self.matrix_orb_index[batch.atom_types].shape,
                    device=orbital_charges.device,
                )
                target_mat_pad[self.matrix_orb_index[batch.atom_types]] = target_charges
                target_charges_diag_pad = torch.gather(
                    target_mat_pad, 1, self.diag_in_matrix_index[batch.atom_types]
                )
                target_charges_diag = target_charges_diag_pad[self.orb_index[batch.atom_types]]

                self.pred_charges_diag = corrected_charges_diag
                self.target_charges_diag = target_charges_diag

            elif self.hparams.charges_type == "atomic_dxtb":
                target_charges = getattr(batch, "atomic_charges_dxtb", None)
                pred_charges = atomic_charges * self.scale_atomic_charges_dxtb
                scale_charges = self.scale_atomic_charges_dxtb
                self.pred_charges_diag = pred_charges
                self.target_charges_diag = target_charges
            elif self.hparams.charges_type == "atomic_lowdin":
                target_charges = getattr(batch, "atomic_charges_lowdin", None)
                pred_charges = atomic_charges * self.scale_atomic_charges_lowdin
                scale_charges = self.scale_atomic_charges_lowdin
                self.pred_charges_diag = pred_charges
                self.target_charges_diag = target_charges
            elif self.hparams.charges_type == "atomic_mulliken":
                target_charges = getattr(batch, "atomic_charges_mulliken", None)
                pred_charges = atomic_charges * self.scale_atomic_charges_mulliken
                scale_charges = self.scale_atomic_charges_mulliken
                self.pred_charges_diag = pred_charges
                self.target_charges_diag = target_charges
        else:
            self.pred_charges_diag = None
            self.target_charges_diag = None

        if getattr(self.hparams, "criterion", "mse") == "mse":
            if (
                loss_coeffs.get("energy", 0.0) > 0.0
                and target_energy is not None
                and pred_energy is not None
            ):
                loss_energy = (
                    (pred_energy / self.scale_energy - target_energy / self.scale_energy)
                    .pow(2)
                    .mean()
                )
                loss += loss_coeffs["energy"] * loss_energy
            if (
                loss_coeffs.get("charges", 0.0) > 0.0
                and pred_charges is not None
                and target_charges is not None
            ):
                loss_charges = (
                    (pred_charges / scale_charges - target_charges / scale_charges).pow(2).mean()
                )
                loss += loss_coeffs["charges"] * loss_charges
            if (
                loss_coeffs.get("forces", 0.0) > 0.0
                and pred_forces is not None
                and hasattr(batch, "forces")
            ):
                target_forces = batch.forces
                loss_forces = (pred_forces - target_forces).pow(2).mean()
                loss += loss_coeffs["forces"] * loss_forces
        elif self.hparams.criterion == "mae":
            if (
                loss_coeffs.get("energy", 0.0) > 0.0
                and target_energy is not None
                and pred_energy is not None
            ):
                loss_energy = (
                    (pred_energy / self.scale_energy - target_energy / self.scale_energy)
                    .abs()
                    .mean()
                )
                loss += loss_coeffs["energy"] * loss_energy
            if (
                loss_coeffs.get("charges", 0.0) > 0.0
                and pred_charges is not None
                and target_charges is not None
            ):
                loss_charges = (
                    (pred_charges / scale_charges - target_charges / scale_charges).abs().mean()
                )
                loss += loss_coeffs["charges"] * loss_charges
            if (
                loss_coeffs.get("forces", 0.0) > 0.0
                and pred_forces is not None
                and hasattr(batch, "forces")
            ):
                target_forces = batch.forces
                loss_forces = (pred_forces - target_forces).abs().mean()
                loss += loss_coeffs["forces"] * loss_forces
        else:
            raise ValueError(f"Unsupported criterion: {self.hparams.criterion}")

        target_forces = getattr(batch, "forces", None)
        return (
            loss,
            pred_energy,
            target_energy,
            pred_charges,
            target_charges,
            pred_forces,
            target_forces,
        )

    def training_step(self, batch, batch_idx):
        (
            loss,
            pred_energy,
            target_energy,
            pred_charges,
            target_charges,
            pred_forces,
            target_forces,
        ) = self(batch)

        log_dict = {}
        if isinstance(loss, torch.Tensor) and loss.requires_grad:
            log_dict["loss/train"] = loss

        if pred_energy is not None and target_energy is not None:
            log_dict["mse/train_energy"] = (pred_energy - target_energy).pow(2).mean().item()
            log_dict["mae/train_energy"] = (pred_energy - target_energy).abs().mean().item()

        if self.pred_charges_diag is not None and self.target_charges_diag is not None:
            log_dict["mse/train_charges"] = (
                (self.pred_charges_diag - self.target_charges_diag).pow(2).mean().item()
            )
            log_dict["mae/train_charges"] = (
                (self.pred_charges_diag - self.target_charges_diag).abs().mean().item()
            )

        if pred_forces is not None and target_forces is not None:
            log_dict["mse/train_forces"] = (pred_forces - target_forces).pow(2).mean().item()
            log_dict["mae/train_forces"] = (pred_forces - target_forces).abs().mean().item()

        if log_dict:
            self.log_dict(log_dict, batch_size=batch["cell"].shape[0], sync_dist=self.distributed)
        return loss

    def validation_step(self, batch, batch_idx):
        (
            loss,
            pred_energy,
            target_energy,
            pred_charges,
            target_charges,
            pred_forces,
            target_forces,
        ) = self(batch)

        log_dict = {}
        if isinstance(loss, torch.Tensor):
            log_dict["loss/val"] = loss

        if pred_energy is not None and target_energy is not None:
            log_dict["mse/val_energy"] = (pred_energy - target_energy).pow(2).mean().item()
            log_dict["mae/val_energy"] = (pred_energy - target_energy).abs().mean().item()

        if self.pred_charges_diag is not None and self.target_charges_diag is not None:
            log_dict["mse/val_charges"] = (
                (self.pred_charges_diag - self.target_charges_diag).pow(2).mean().item()
            )
            log_dict["mae/val_charges"] = (
                (self.pred_charges_diag - self.target_charges_diag).abs().mean().item()
            )

        if pred_forces is not None and target_forces is not None:
            log_dict["mse/val_forces"] = (pred_forces - target_forces).pow(2).mean().item()
            log_dict["mae/val_forces"] = (pred_forces - target_forces).abs().mean().item()

        if log_dict:
            self.log_dict(log_dict, batch_size=batch["cell"].shape[0], sync_dist=self.distributed)

        ret_dict = {
            "n_atom": batch.n_atom,
            "atom_types": batch.atom_types,
            "coords": batch.coords,
        }
        if isinstance(loss, torch.Tensor):
            ret_dict["loss"] = loss

        if pred_energy is not None and target_energy is not None:
            ret_dict["pred_energy"] = pred_energy
            ret_dict["target_energy"] = target_energy
        if self.pred_charges_diag is not None and self.target_charges_diag is not None:
            ret_dict["pred_charges"] = self.pred_charges_diag
            ret_dict["target_charges"] = self.target_charges_diag
        if pred_forces is not None and target_forces is not None:
            ret_dict["pred_forces"] = pred_forces
            ret_dict["target_forces"] = target_forces

        return ret_dict

    def test_step(self, batch, batch_idx):
        (
            loss,
            pred_energy,
            target_energy,
            pred_charges,
            target_charges,
            pred_forces,
            target_forces,
        ) = self(batch)

        log_dict = {}
        if isinstance(loss, torch.Tensor):
            log_dict["loss/test"] = loss

        if pred_energy is not None and target_energy is not None:
            log_dict["mse/test_energy"] = (pred_energy - target_energy).pow(2).mean().item()
            log_dict["mae/test_energy"] = (pred_energy - target_energy).abs().mean().item()

        if self.pred_charges_diag is not None and self.target_charges_diag is not None:
            log_dict["mse/test_charges"] = (
                (self.pred_charges_diag - self.target_charges_diag).pow(2).mean().item()
            )
            log_dict["mae/test_charges"] = (
                (self.pred_charges_diag - self.target_charges_diag).abs().mean().item()
            )

        if pred_forces is not None and target_forces is not None:
            log_dict["mse/test_forces"] = (pred_forces - target_forces).pow(2).mean().item()
            log_dict["mae/test_forces"] = (pred_forces - target_forces).abs().mean().item()

        if log_dict:
            self.log_dict(log_dict, batch_size=batch["cell"].shape[0], sync_dist=self.distributed)
        return loss

    def configure_optimizers(self):
        opt = instantiate(
            self.hparams.train.optim,
            params=self.parameters(),
            _convert_="partial",
        )
        scheduler = instantiate(self.hparams.train.lr_scheduler, optimizer=opt)

        if "lr_schedule_freq" in self.hparams.train:
            scheduler = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": self.hparams.train.lr_schedule_freq,
                "monitor": self.hparams.train.monitor.metric,
            }

        return {
            "optimizer": opt,
            "lr_scheduler": scheduler,
            "monitor": self.hparams.train.monitor.metric,
        }

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
            log.warning(
                "Failed to load EMA state dict (%s). Please make sure this was intended.", e
            )

    def on_validation_epoch_start(self):
        self.ema.store()
        self.ema.copy_to(self.parameters())

    def on_validation_epoch_end(self):
        self.ema.restore()
        if isinstance(self.lr_schedulers(), torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_schedulers().step(
                self.trainer.callback_metrics[self.hparams.train.monitor.metric]
            )

    def on_before_zero_grad(self, optimizer):
        self.ema.update(self.parameters())

    def on_after_backward(self):
        total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), float("inf"), norm_type=2.0)
        self.log("trainer/grad_norm", total_norm)

    def freeze_backbone(self):
        log.info("Freezing backbone (all layers except EnergyBlock)...")
        for param in self.model.parameters():
            param.requires_grad = False

        found_energy_block = False
        for name, module in self.model.named_modules():
            if "EnergyBlock" in module.__class__.__name__:
                log.info("Unfreezing module: %s (%s)", name, module.__class__.__name__)
                for param in module.parameters():
                    param.requires_grad = True
                found_energy_block = True

        if not found_energy_block:
            log.warning("EnergyBlock not found! All parameters are frozen.")
