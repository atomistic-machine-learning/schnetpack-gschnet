import argparse
from pathlib import Path
from ase.db import connect
from ase import Atoms
import numpy as np
import logging
import pickle
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_parser():
    """Setup parser for command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("datapath", help="Path to data base")
    parser.add_argument(
        "--overwrite",
        help="Overwrite the existing data base with the ordered data base",
        action="store_true",
    )
    parser.add_argument(
        "--write_mol_dict", help="Write a mol_dict file", action="store_true"
    )
    parser.add_argument(
        "--atom_types", help="Atom types used to sort db", default=[6, 7, 8, 9, 1]
    )
    return parser


def sort_db(args):
    source_db = Path(args.datapath)
    if not source_db.exists():
        log.error(f"There is no data base at {source_db}.")
        raise FileNotFoundError
    with connect(source_db) as source:
        n_mols = source.count()
        if n_mols == 0:
            log.error(f"There are no molecules in the data base at {source_db}")
            raise ValueError
    if args.overwrite:
        target_db = source_db
    else:
        target_db = source_db.with_name(source_db.stem + "_sorted.db")
        if target_db.exists():
            with connect(target_db) as con:
                if con.count() == n_mols:
                    log.info("Sorted db does already exist! Exiting...")
                    return
                elif con.count() > 0:
                    log.error(
                        f"Sorted db {target_db} does already exist but contains "
                        f"{con.count()} molecules, whereas the input db contains "
                        f"{n_mols} molecules. Aborting."
                    )
                    raise FileExistsError
    atom_types = np.array(args.atom_types)
    max_type = max(atom_types)

    log.info("Reading molecules from data base...")
    gathered_mols = {}
    with connect(source_db) as source:
        for i in tqdm(range(source.count())):
            row = source.get(i + 1)
            pos = row.positions
            numbers = row.numbers
            update_dict(gathered_mols, pos, numbers, max_type)
    log.info("...done!")

    log.info("Sorting molecules...")
    sorted_gathered_mols = {}
    for key in tqdm(gathered_mols):
        d = gathered_mols[key]
        positions = np.array(d["positions"])
        numbers = np.array(d["numbers"])
        n_types = np.array(d["n_types"])
        order = np.lexsort(n_types.T[::-1])
        sorted_gathered_mols[key] = {
            "_positions": positions[order],
            "_atomic_numbers": numbers[order],
        }
    del gathered_mols
    log.info("...done!")

    log.info(f"Writing molecules to data base {target_db}...")
    if target_db == source_db:
        source_db.unlink(missing_ok=False)
    with connect(target_db) as target:
        with tqdm(total=n_mols) as pbar:
            for key in sorted(sorted_gathered_mols):
                positions = sorted_gathered_mols[key]["_positions"]
                numbers = sorted_gathered_mols[key]["_atomic_numbers"]
                for index in range(len(numbers)):
                    at = Atoms(
                        numbers=numbers[index],
                        positions=positions[index],
                    )
                    target.write(at)
                    pbar.update()
    log.info("... done!")

    if args.write_mol_dict:
        mol_dict = target_db.with_suffix(".mol_dict")
        log.info("Writing .mol_dict file...")
        with open(mol_dict, "wb") as f:
            pickle.dump(sorted_gathered_mols, f)
        log.info("... done!")


def update_dict(d, pos, numbers, max_type):
    n_atoms = len(pos)
    if n_atoms in d:
        d[n_atoms]["positions"] += [pos]
        d[n_atoms]["numbers"] += [numbers]
        d[n_atoms]["n_types"] += [np.bincount(numbers, minlength=max_type + 1)]
    else:
        d[n_atoms] = {
            "positions": [pos],
            "numbers": [numbers],
            "n_types": [np.bincount(numbers, minlength=max_type + 1)],
        }
    return d


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    sort_db(args)
