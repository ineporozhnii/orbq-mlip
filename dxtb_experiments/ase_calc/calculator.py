from typing import List, Optional

import torch
import ase.calculators.calculator as ase_calc
from ase.atoms import Atoms
from ase.units import Hartree, Bohr
from dxtb.calculators import GFN1Calculator, GFN2Calculator


class DXTB(ase_calc.Calculator):
    """ASE calculator for xtb related methods.

    The XTB class can access all methods exposed by the ``xtb`` API.
    """

    implemented_properties = [
        "energy",
        "forces",
        "charges",
    ]

    default_parameters = {
        "method": "GFN2-xTB",
        "accuracy": 1.0,
        "max_iterations": 250,
        "electronic_temperature": 300.0,
        "solvent": "None",
        "cache_api": True,
        "verbosity": 0,
    }

    _res = None
    _dxtb = None

    def __init__(
        self, atoms: Optional[Atoms] = None, **kwargs,
    ):
        """Construct the dxtb base calculator object."""

        ase_calc.Calculator.__init__(self, atoms=atoms, **kwargs)

    def set(self, **kwargs) -> dict:
        """Set new parameters to dxtb"""

        changed_parameters = ase_calc.Calculator.set(self, **kwargs)

        self._check_parameters(changed_parameters)

        # Always reset the calculation if parameters change
        if changed_parameters:
            self.reset()

        # If the method is changed, invalidate the cached calculator as well
        if "method" in changed_parameters:
            self._dxtb = None
            self._res = None

        # Minor changes can be updated in the API calculator directly
        if self._dxtb is not None:
            if "accuracy" in changed_parameters:
                self._dxtb.set_accuracy(self.parameters.accuracy)

            if "electronic_temperature" in changed_parameters:
                self._dxtb.set_electronic_temperature(
                    self.parameters.electronic_temperature
                )

            if "max_iterations" in changed_parameters:
                self._dxtb.set_max_iterations(self.parameters.max_iterations)

        return changed_parameters

    def _check_parameters(self, parameters: dict) -> None:
        """Verifiy provided parameters are valid"""

        if "method" in parameters and parameters["method"] not in ["GFN1-xTB", "GFN2-xTB"]:
            raise ase_calc.InputError(
                "Invalid method {} provided".format(parameters["method"])
            )

    def reset(self) -> None:
        """Clear all information from old calculation"""
        ase_calc.Calculator.reset(self)

        if not self.parameters.cache_api:
            self._dxtb = None
            self._res = None

    def _check_api_calculator(self, system_changes: List[str]) -> None:
        """Check state of API calculator and reset if necessary"""

        # Changes in positions and cell parameters can use a normal update
        _reset = system_changes.copy()
        if "positions" in _reset:
            _reset.remove("positions")
        if "cell" in _reset:
            _reset.remove("cell")

        # Invalidate cached calculator and results object
        if _reset:
            self._dxtb = None
            self._res = None
        else:
            if system_changes and self._dxtb is not None:
                try:
                    _cell = self.atoms.cell
                    self._dxtb.update(
                        self.atoms.positions / Bohr, _cell / Bohr,
                    )
                # An exception in this part means the geometry is bad,
                # still we will give a complete reset a try as well
                except Exception as e:
                    self._dxtb = None
                    self._res = None

    def _create_api_calculator(self) -> GFN2Calculator:
        """Create a new API calculator object"""

        _method = self.parameters.method 
        if _method is None:
            raise ase_calc.InputError(
                "Invalid method {} provided".format(self.parameters.method)
            )

        try:
            if _method == "GFN1-xTB":
                calc = GFN1Calculator(torch.tensor(self.atoms.numbers, dtype=torch.int64), 
                                      dtype = torch.double, 
                                      device = torch.device("cpu"),
                                      opts={"maxiter": self.parameters.max_iterations, "verbosity": self.parameters.verbosity})
            elif _method == "GFN2-xTB":
                calc = GFN2Calculator(torch.tensor(self.atoms.numbers, dtype=torch.int64), 
                                      dtype = torch.double, 
                                      device = torch.device("cpu"),
                                      opts={"maxiter": self.parameters.max_iterations, "verbosity": self.parameters.verbosity})

        except Exception as e:
            print(e)
            raise ase_calc.InputError("Cannot construct calculator for dxtb")

        return calc

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: List[str] = None,
        system_changes: List[str] = ase_calc.all_changes,
    ) -> None:
        """Perform actual calculation with by calling the xtb API"""

        if not properties:
            properties = ["energy"]
        ase_calc.Calculator.calculate(self, atoms, properties, system_changes)

        #self._check_api_calculator(system_changes)

        if self._dxtb is None:
            self._dxtb = self._create_api_calculator()

        try:
            pos = torch.tensor(self.atoms.positions  / Bohr, dtype=torch.double).requires_grad_(True)
            self._res = self._dxtb.singlepoint(pos,
                                               chrg=self.atoms.get_initial_charges().sum(),
                                               initial_state=None)
        except Exception as e:
            raise ase_calc.CalculationFailed("dxtb could not evaluate input")

        # These properties are garanteed to exist for all implemented calculators
        self.results["energy"] = self._res.total.sum().detach().numpy() * Hartree
        self.results["free_energy"] = self.results["energy"]
        
        self.results["forces"] = -torch.autograd.grad(self._res.total.sum(), pos)[0].detach().numpy() * Hartree / Bohr
        self.results["charges"] = self._res.charges.mono.detach().numpy()