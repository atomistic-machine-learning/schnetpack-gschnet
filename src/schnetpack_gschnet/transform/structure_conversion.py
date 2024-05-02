from typing import Dict, Optional, List
import logging
from io import StringIO
import torch
from schnetpack.transform.base import Transform
from schnetpack_gschnet import properties

from rdkit.rdBase import BlockLogs
from rdkit import Chem
from rdkit.Chem import (
    rdDetermineBonds,
    rdMolDescriptors,
    AtomValenceException,
    Descriptors,
)
from ase import Atoms
from ase.io import write

logger = logging.getLogger(__name__)

__all__ = [
    "GetSmiles",
]


class GetSmiles(Transform):
    """
    Get the canonical smiles of the 3d structure using RDKit.
    If the argument `return_inputs` is set, the smiles string is encoded in
    ascii bytes and stored in inputs as a uint8 tensor under the key `smiles`
    along with an entry `n_smiles_bytes` (ready for batching).
    Otherwise, a dictionary is returned with the smiles as string under the key
    'smiles'.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(
        self,
        allowed_charges: Optional[List[int]] = [0],
        allow_charged_fragments: Optional[bool] = False,
        allow_radical_electrons: Optional[bool] = False,
        return_inputs: Optional[bool] = True,
        store_validity: Optional[bool] = False,
        store_chirality_smiles: Optional[bool] = False,
        store_chemical_formula: Optional[bool] = False,
        store_ring_statistics: Optional[bool] = False,
    ):
        """
        Args:
            allowed_charges:
            allow_charged_fragments:
            allow_radical_electrons:
            return_inputs:
            store_validity:
            store_chirality_smiles:
            store_chemical_formula:
            store_ring_statistics:
        """
        super().__init__()
        self.allowed_charges = allowed_charges
        self.allow_charged_fragments = allow_charged_fragments
        self.allow_radical_electrons = allow_radical_electrons
        self.return_inputs = return_inputs
        self.store_validity = store_validity
        self.store_chirality_smiles = store_chirality_smiles
        self.store_chemical_formula = store_chemical_formula
        self.store_ring_statistics = store_ring_statistics
        if not self.allow_charged_fragments and 0 not in self.allowed_charges:
            raise ValueError(
                f"You specified `allow_charged_fragments`="
                f"{self.allow_charged_fragments} and `allowed_charges`="
                f"{self.allowed_charges}. However, the total charge can "
                f"only be 0 if no charged fragments are allowed."
            )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # create rdkit Mol via string with xyz block
        positions = inputs[properties.R]
        numbers = inputs[properties.Z]
        atoms = Atoms(numbers=numbers, positions=positions)
        tmp = StringIO()
        write(tmp, atoms, format="xyz")
        tmp.seek(0)
        xyz = "".join(tmp)
        mol = Chem.MolFromXYZBlock(xyz)
        formula = ""
        smiles = ""
        block = BlockLogs()  # block logs from rdkit
        if not self.allow_charged_fragments:
            # total charge can only be 0 without charged fragments
            allowed_charges = [
                0,
            ]
        else:
            allowed_charges = self.allowed_charges
        for charge in allowed_charges:
            try:
                rdDetermineBonds.DetermineBonds(
                    mol,
                    charge=0,
                    allowChargedFragments=self.allow_charged_fragments,
                    embedChiral=self.store_chirality_smiles,
                )
                mol = Chem.RemoveHs(mol)
                if (
                    not self.allow_charged_fragments
                    and not self.allow_radical_electrons
                ):
                    # check for radical electrons
                    radical_found = False
                    for at in mol.GetAtoms():
                        if at.GetNumRadicalElectrons() > 0:
                            radical_found = True
                            break
                    if radical_found:
                        break
                formula = rdMolDescriptors.CalcMolFormula(mol)
                smiles = Chem.MolToSmiles(mol)
                smiles = Chem.CanonSmiles(smiles)
                # stop if a smiles for the current charge was found
                break
            except AtomValenceException:
                continue
            except ValueError:
                continue
        del block  # enable logs from rdkit
        results = {"smiles": smiles}
        if self.store_validity:
            if smiles == "" or "." in smiles:
                # invalid if no smiles could be found or multiple fragments exist
                results["validity"] = False
            else:
                results["validity"] = True
        if self.store_chirality_smiles:
            if "@" in smiles:
                results["smiles"] = smiles.replace("@", "")
                results["chirality_smiles"] = smiles
            else:
                results["chirality_smiles"] = ""
        if self.store_chemical_formula:
            results["formula"] = formula
        if self.store_ring_statistics:
            if smiles == "":
                results["R3"] = 0
                results["R4"] = 0
                results["R5"] = 0
                results["R6"] = 0
            else:
                sssr = Chem.GetSymmSSSR(mol)
                rings = [len(sssr[i]) for i in range(len(sssr))]
                n_rings = torch.bincount(
                    torch.tensor(rings, dtype=torch.int), minlength=7
                )
                results["R3"] = n_rings[3]
                results["R4"] = n_rings[4]
                results["R5"] = n_rings[5]
                results["R6"] = n_rings[6]
        if self.return_inputs:
            # all entries in inputs need to be of type torch.Tensor to be batchable
            for key in results:
                if isinstance(results[key], str):
                    # there is no str tensor, we have to convert the string instead
                    # encode as ascii bytes and store in uint8 tensor
                    ascii_bytes = list(bytes(results[key], "ascii"))
                    inputs[key] = torch.tensor(ascii_bytes, dtype=torch.uint8)
                    inputs[f"n_{key}_bytes"] = torch.tensor(
                        [len(ascii_bytes)], dtype=torch.int
                    )
                else:
                    inputs[key] = torch.tensor([results[key]])
            return inputs
        else:
            return results
