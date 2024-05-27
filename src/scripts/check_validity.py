import argparse
from pathlib import Path

import torch.utils.data.sampler
from ase.db import connect
from ase import Atoms
import numpy as np
import logging
from tqdm import tqdm
from rdkit import Chem

from schnetpack.data import load_dataset, AtomsDataFormat, AtomsLoader
from schnetpack_gschnet.transform import GetSmiles

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_parser():
    """Setup parser for command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to data base with molecules.")
    parser.add_argument(
        "--results_path",
        help="The computed statistics are stored as an npz file at this path "
        "if provided. "
        "Set `--results_path auto` to automatically generate the path "
        "from `data_path` by appending '_validity_statistics.npz' to the data "
        "base filename."
        "CAUTION: If the file already exists, the statistics are instead"
        "loaded from that file. In order to force a recomputation, set"
        "`--force_recomputation` (which will overwrite the existing file).",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--timeout",
        help="A timeout in seconds. If the validity check cannot be completed "
        "within this timeframe, the molecule is marked as invalid. Set to "
        "`0` to disable timeouts. BEWARE: Some checks can take immense amounts "
        "of time, therefore we recommend to set a timeout. (Default: 5)",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--allowed_charges",
        help="List of allowed total charges of the molecules, where the charges are "
        "tested in the order they appear in the list. As soon as a valid "
        "smiles for a charge has been found, the remaining charges in the list "
        "are ommited, i.e. only the first found smiles is used. "
        "CAUTION: Charges other than 0 are only considered if "
        "`--allow_charged_fragments` is set because otherwise the total charge "
        "can only be 0. (Default: [0,])",
        type=int,
        nargs="+",
        default=[
            0,
        ],
    )
    parser.add_argument(
        "--allow_charged_fragments",
        help="Allow charged fragments (i.e. formal charges will be placed on atoms "
        "according to their valency). Otherwise, radical electrons will be "
        "placed on the atoms.",
        action="store_true",
    )
    parser.add_argument(
        "--allow_radical_electrons",
        help="Allow radical electrons. Otherwise, structures where radical "
        "electrons have been placed (only happens if "
        "`--alow_charged_fragments` is not specified) are counted as "
        "invalid.",
        action="store_true",
    )
    parser.add_argument(
        "--compute_ring_statistics",
        help="Compute the number of 3-, 4-, 5-, and 6-membered rings using "
        "the symmetric SSSR from RDKit.",
        action="store_true",
    )
    parser.add_argument(
        "--compute_uniqueness",
        help="If set, also computes/stores the uniqueness of molecules, i.e. all "
        "smiles strings are compared with each other to identify duplicate "
        "molecules.",
        action="store_true",
    )
    parser.add_argument(
        "--ignore_enantiomers",
        help="If set, mirror-image stereoisomers (enantiomers) are classified as "
        "being identical structures when checking the uniqueness of molecules "
        "or comparing them to training structures.",
        action="store_true",
    )
    parser.add_argument(
        "--ignore_isomerism",
        help="If set, all isomeric information in smiles strings is ignored when "
        "checking the uniqueness of molecules or comparing them to training "
        "structures (i.e. we use the non-isomeric smiles for comparison). "
        "However, the stored smiles strings will always be isomeric.",
        action="store_true",
    )
    parser.add_argument(
        "--compare_db_path",
        help="Compare the smiles of valid molecules with smiles of molecules from "
        "this data base (e.g. the training data base). If there is a field "
        "`smiles` for molecules in the db, the stored smiles are used instead "
        "of computing them.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--compare_db_split_path",
        help="Split file that tells which molecules in the data base provided with "
        "`--compare_db_path` were training, validation, and test structures. "
        "The comparison will only be made with respect to these molecules, "
        "i.e. molecules which are in neither of the three sets will be "
        "ommited.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--compare_db_results_path",
        help="If the file already exists, the smiles are loaded from here instead of "
        "computed. If the file does not exist, the smiles are stored here after "
        "they have been computed."
        "Set `--compare_db_results_path auto` to automatically generate the path "
        "from `--compare_db_path` by appending '_validity_statistics.npz' to the "
        "data base filename.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--force_recomputation",
        help="If set, the statistics are recomputed and the file at "
        "`--results_path` is overwritten.",
        action="store_true",
    )
    parser.add_argument(
        "--write_to_db",
        help="If set, the validity value (True or False) and the obtained smiles "
        "string are written to the input data base. "
        "The validity can be found as row.data['validity'] while the smiles is "
        "stored under row.smiles. "
        "CAUTION: Existing values of `data['validity']` and `smiles` in the db "
        "are overwritten. Furthermore, other values like the ring statistics "
        "or the uniqueness are not written to the data base.",
        action="store_true",
    )
    return parser


def main(
    data_path,
    results_path=None,
    allowed_charges=[
        0,
    ],
    allow_charged_fragments=False,
    allow_radical_electrons=False,
    compute_ring_statistics=False,
    force_recomputation=False,
    ignore_enantiomers=False,
    ignore_isomerism=False,
    compute_uniqueness=False,
    compare_db_path=None,
    compare_db_results_path=None,
    compare_db_split_path=None,
    timeout=5,
    write_to_db=False,
    ignore_warnings=False,
):
    db_path = Path(data_path)
    if results_path is not None and results_path.lower() == "auto":
        results_path = db_path.with_name(db_path.stem + "_validity_statistics.npz")
    if results_path is None or not Path(results_path).exists() or force_recomputation:
        # compute statistics for data base
        # setup
        dataset = load_dataset(
            db_path,
            AtomsDataFormat.ASE,
            load_properties=[],
            load_structure=True,
        )
        dataset.transforms = [
            GetSmiles(
                allowed_charges=allowed_charges,
                allow_charged_fragments=allow_charged_fragments,
                allow_radical_electrons=allow_radical_electrons,
                return_inputs=False,
                store_validity=True,
                store_chemical_formula=True,
                store_ring_statistics=compute_ring_statistics,
            )
        ]
        _batch_size = 1
        sampler = SamplerWithSkipping(dataset)
        loader = AtomsLoader(
            dataset,
            batch_size=_batch_size,
            num_workers=1,
            sampler=sampler,
            pin_memory=False,
            collate_fn=lambda x: x,
            timeout=timeout,
        )
        n_mols = len(dataset)
        if compute_ring_statistics:
            numeric_stats = np.zeros((n_mols, 5), dtype=int)
        else:
            numeric_stats = np.zeros((n_mols, 1), dtype=int)
        smiles = [""] * n_mols
        formulas = [""] * n_mols
        i = 0
        failed_i = 0

        # check validity by computing smiles
        print(
            f"\nComputing smiles strings and checking their validity for "
            f"{n_mols} molecules from {db_path}..."
        )
        with tqdm(total=n_mols) as pbar:
            while i < n_mols:
                i = validity_check_loop(
                    loader=loader,
                    pbar=pbar,
                    batch_size=_batch_size,
                    numeric_stats=numeric_stats,
                    smiles=smiles,
                    formulas=formulas,
                    compute_ring_statistics=compute_ring_statistics,
                )
                if i < n_mols:
                    # an exception stopped the loop, one molecule was skipped
                    sampler.skip_steps(i)
                    failed_i += 1
        if failed_i > 0:
            print(
                f"\nATTENTION: {failed_i} molecules were marked as invalid due "
                f"to a timeout during the validity check. The current timeout is "
                f"{timeout} seconds. Set a higher value with `--timeout` to "
                f"potentially reduce the number of timeouts. Setting `--timeout 0` "
                f"disables the timeout, however, this is not recommended as some "
                f"checks can take an immense amount of time."
            )
        validity = numeric_stats[:, 0]

        # check uniqueness of smiles
        if compute_uniqueness:
            print(f"\nChecking uniqueness of valid molecules...")
            unique, n_duplicates, duplicating, smiles_lookup = uniqueness_check_loop(
                validity=validity,
                smiles=smiles,
                formulas=formulas,
                ignore_enantiomers=ignore_enantiomers,
                ignore_isomerism=ignore_isomerism,
            )
            numeric_stats = np.hstack(
                (
                    numeric_stats,
                    unique[:, None],
                    n_duplicates[:, None],
                    duplicating[:, None],
                )
            )
        else:
            unique = None
            duplicating = None

        # compare to other db (e.g. training data)
        if compare_db_path is not None:
            print(
                f"\nComparing smiles of valid molecules to smiles of molecules "
                f"from the data base at {compare_db_path}."
            )
            known, known_idx, known_split = compare_with_db(
                validity=validity,
                smiles=smiles,
                formulas=formulas,
                unique=unique,
                duplicating=duplicating,
                compare_db_path=compare_db_path,
                compare_db_results_path=compare_db_results_path,
                compare_db_split_path=compare_db_split_path,
                allowed_charges=allowed_charges,
                allow_charged_fragments=allow_charged_fragments,
                allow_radical_electrons=allow_radical_electrons,
                ignore_enantiomers=ignore_enantiomers,
                ignore_isomerism=ignore_isomerism,
                timeout=timeout,
            )
            numeric_stats = np.hstack(
                (
                    numeric_stats,
                    known[:, None],
                    known_idx[:, None],
                )
            )
            if known_split is not None:
                numeric_stats = np.hstack((numeric_stats, known_split[:, None]))

        # store results
        if results_path is not None:
            print(
                f"\nStoring calculated results for {data_path} " f"at {results_path}."
            )
            results_path = Path(results_path)
            numeric_columns = ["validity"]
            if compute_ring_statistics:
                numeric_columns += ["R3", "R4", "R5", "R6"]
            if compute_uniqueness:
                numeric_columns += ["uniqueness", "n_duplicates", "duplicating"]
            if compare_db_path is not None:
                numeric_columns += ["known", "known_idx"]
                if known_split is not None:
                    numeric_columns += ["known_split"]
            string_statistics = [formulas, smiles]
            string_columns = ["chemical_formula", "smiles"]
            string_statistics = np.array(string_statistics).T
            np.savez(
                results_path,
                numeric_statistics=numeric_stats,
                numeric_columns=numeric_columns,
                string_statistics=string_statistics,
                string_columns=string_columns,
            )
            print(
                f"The npz file contains the arrays `numeric_statistics` of shape "
                f"{numeric_stats.shape} and `string_statistics` of shape "
                f"{string_statistics.shape}. The column names have also been stored "
                f"in the file as `numeric_columns`={numeric_columns} and "
                f"`string_columns`={string_columns}."
            )
    else:
        # load existing statistics
        if not ignore_warnings:
            print(
                f"\nATTENTION: File {results_path} already exists. Reading results "
                f"from this file.\nPlease set another path for `--results_path` or "
                f"use `--force_recomputation` if you want to recompute the "
                f"statistics and overwrite the existing file.\n"
            )
        results = np.load(results_path)
        numeric_stats = results["numeric_statistics"]
        string_stats = results["string_statistics"]
        numeric_columns = results["numeric_columns"]
        string_columns = results["string_columns"]
        idx = lambda x, y: np.argwhere(x == y)[0, 0]
        validity = numeric_stats[:, idx(numeric_columns, "validity")]
        formulas = string_stats[:, idx(string_columns, "chemical_formula")]
        smiles = string_stats[:, idx(string_columns, "smiles")]
        n_mols = len(validity)
        if "uniqueness" not in numeric_columns:
            if compute_uniqueness:
                print(f"\nChecking uniqueness of valid molecules...")
                (
                    unique,
                    n_duplicates,
                    duplicating,
                    smiles_lookup,
                ) = uniqueness_check_loop(
                    validity=validity,
                    smiles=smiles,
                    formulas=formulas,
                    ignore_enantiomers=ignore_enantiomers,
                    ignore_isomerism=ignore_isomerism,
                )
            else:
                unique = None
                duplicating = None
        else:
            unique = numeric_stats[:, idx(numeric_columns, "uniqueness")]
            n_duplicates = numeric_stats[:, idx(numeric_columns, "n_duplicates")]
            duplicating = numeric_stats[:, idx(numeric_columns, "duplicating")]
        if compare_db_path is not None:
            if "known_idx" not in numeric_columns or (
                compare_db_split_path is not None
                and "known_split" not in numeric_columns
            ):
                print(
                    f"\nComparing smiles of valid molecules to smiles of molecules "
                    f"from the data base at {compare_db_path}."
                )
                known, known_idx, known_split = compare_with_db(
                    validity=validity,
                    smiles=smiles,
                    formulas=formulas,
                    unique=unique,
                    duplicating=duplicating,
                    compare_db_path=compare_db_path,
                    compare_db_results_path=compare_db_results_path,
                    compare_db_split_path=compare_db_split_path,
                    allowed_charges=allowed_charges,
                    allow_charged_fragments=allow_charged_fragments,
                    allow_radical_electrons=allow_radical_electrons,
                    ignore_enantiomers=ignore_enantiomers,
                    ignore_isomerism=ignore_isomerism,
                    timeout=timeout,
                )
            else:
                known = numeric_stats[:, idx(numeric_columns, "known")]
                if "known_split" in numeric_columns:
                    known_split = numeric_stats[:, idx(numeric_columns, "known_split")]
                known_idx = numeric_stats[:, idx(numeric_columns, "known_idx")]

    n_valid = np.sum(validity)
    print(f"\n{n_valid} molecules of {n_mols} are valid.")
    if compute_uniqueness:
        n_unique = np.sum(unique)
        print(f"Of these, {n_unique} molecules are unique.")
        if compare_db_path is not None:
            # exclude non-unique
            known = known[unique.astype(bool)]
            if compare_db_split_path is not None:
                known_split = known_split[unique.astype(bool)]
    if compare_db_path is not None:
        if compare_db_split_path is not None:
            n_novel = np.sum(known_split == 0)
            n_train = np.sum(known_split == 1)
            n_val = np.sum(known_split == 2)
            n_test = np.sum(known_split == 3)
            print(
                f"Of these, {n_train} (training) + {n_val} (validation) = "
                f"{n_train + n_val} structures resemble molecules that were used "
                f"for training."
                f"\n{n_test} (test) + {n_novel} (novel) = {n_test + n_novel} "
                f"structures are unseen molecules that were not used for training."
            )
        else:
            n_novel = np.sum(known == 0)
            n_known = np.sum(known == 1)
            print(
                f"Of these, {n_known} molecules resemble structures from "
                f"{compare_db_path}."
                f"\n{n_novel} molecules were not found in that data base."
            )

    if write_to_db:
        print(f"\nWriting validity values of molecules to db at {db_path}.")
        with connect(db_path) as con:
            md = con.metadata
            if "validity" not in md["_property_unit_dict"]:
                md["_property_unit_dict"].update({"validity": "bool"})
                con.metadata = md
            for i in tqdm(range(con.count())):
                row = con.get(i + 1)
                data = row.data
                data["validity"] = np.array([validity[i]], dtype=bool)
                con.update(row.id, data=data, smiles=smiles[i])

    return validity, smiles, formulas


def validity_check_loop(
    loader,
    pbar,
    batch_size,
    numeric_stats,
    smiles,
    formulas,
    compute_ring_statistics,
):
    i = loader.sampler.skip
    try:
        for result in loader:
            for r in result:
                numeric_stats[i, 0] = r["validity"]
                if compute_ring_statistics:
                    numeric_stats[i, 1] = r["R3"]
                    numeric_stats[i, 2] = r["R4"]
                    numeric_stats[i, 3] = r["R5"]
                    numeric_stats[i, 4] = r["R6"]
                smiles[i] = r["smiles"]
                formulas[i] = r["formula"]
                i += 1
            pbar.update(batch_size)
    except RuntimeError:
        i += 1
        pbar.update(batch_size)
    return i


def uniqueness_check_loop(
    validity,
    smiles,
    formulas,
    ignore_enantiomers,
    ignore_isomerism,
):
    unique = np.zeros(len(validity), dtype=bool)
    duplicating = np.ones(len(validity), dtype=int) * -1
    n_duplicates = np.zeros(len(validity), dtype=int)
    smiles_lookup = {}
    for i in tqdm(range(len(validity))):
        if validity[i]:
            f = formulas[i]
            s = smiles[i]
            if ignore_isomerism:
                # remove isomerism from smiles
                s = Chem.CanonSmiles(s, useChiral=False)
            if f not in smiles_lookup:
                # molecule is unique
                if ignore_enantiomers and not ignore_isomerism:
                    # also add smiles of mirror-image stereoisomer
                    s2 = s.replace("@", "@@").replace("@@@@", "@")
                    smiles_lookup[f] = {s: i, s2: i}
                else:
                    smiles_lookup[f] = {s: i}
                unique[i] = 1
            elif s not in smiles_lookup[f]:
                # molecule is unique
                if ignore_enantiomers and not ignore_isomerism:
                    # also add smiles of mirror-image stereoisomer
                    s2 = s.replace("@", "@@").replace("@@@@", "@")
                    smiles_lookup[f].update({s: i, s2: i})
                else:
                    smiles_lookup[f].update({s: i})
                unique[i] = 1
            else:
                # molecule is not unique
                unique_idx = smiles_lookup[f][s]
                duplicating[i] = unique_idx
                n_duplicates[unique_idx] += 1
    return unique, n_duplicates, duplicating, smiles_lookup


def compare_with_db(
    validity,
    smiles,
    formulas,
    unique,
    duplicating,
    compare_db_path,
    compare_db_results_path,
    compare_db_split_path,
    allowed_charges,
    allow_charged_fragments,
    allow_radical_electrons,
    ignore_enantiomers,
    ignore_isomerism,
    timeout,
):
    if not Path(compare_db_path).exists():
        raise ValueError(
            f"There exists no data base at the path provided "
            f"for comparison (`--compare_db_path {compare_db_path})."
        )
    with connect(compare_db_path) as con:
        db_smiles = []
        db_formulas = []
        db_validity = np.zeros(con.count(), dtype=bool)
        if con.count() == 0:
            raise ValueError(
                f"There are no molecules in the data base at the path "
                f"provided for comparison (`--compare_db_path "
                f"{compare_db_path})."
            )
        if (
            compare_db_results_path is None
            or not Path(compare_db_results_path).exists()
        ):
            # try to read smiles from db
            row = con.get(1)
            if "smiles" in row:
                print(
                    f"Reading smiles from data base specified for comparison "
                    f"({compare_db_path})."
                )
                for i in tqdm(range(con.count())):
                    row = con.get(i + 1)
                    db_smiles += [row.smiles]
                    db_validity[i] = db_smiles[i] != ""
                    db_formulas += [row.toatoms().get_chemical_formula()]
    if len(db_smiles) == 0:  # did not read smiles from db
        # compute validity, smiles, and formula
        db_validity, db_smiles, db_formulas = main(
            data_path=compare_db_path,
            results_path=compare_db_results_path,
            allowed_charges=allowed_charges,
            allow_charged_fragments=allow_charged_fragments,
            allow_radical_electrons=allow_radical_electrons,
            timeout=timeout,
            ignore_warnings=True,
        )
    # load split if provided
    if compare_db_split_path is not None:
        if not Path(compare_db_split_path).exists():
            raise ValueError(
                f"The provided split file ({compare_db_split_path}) does not exist."
            )
        split = np.load(compare_db_split_path)
        train_set = split["train_idx"]
        val_set = split["val_idx"]
        test_set = split["test_idx"]
        if "placement_cutoff" in split:
            key = (
                f"{split['placement_cutoff']}/{split['use_covalent_radii']}"
                f"/{split['covalent_radius_factor']}"
            )
            with connect(compare_db_path) as con:
                # look for subset in data base
                md = con.metadata
                if "disconnected_idx" not in md or key not in md["disconnected_idx"]:
                    raise RuntimeError(
                        f"Cannot match indices provided in the split file with "
                        f"indices from the data base. The provided split file "
                        f"({compare_db_split_path}) was sampled with the setting "
                        f"{key} (placement_cutoff/use_covalent_radii/"
                        f"covalent_radius_factor). However, the setting cannot be "
                        f"found in the metadata of the data base "
                        f"({compare_db_path})."
                    )
                excluded = md["disconnected_idx"][key]
                mask = np.ones(len(db_validity), dtype=bool)
                mask[excluded] = 0
                index_map = np.nonzero(mask)[0]
                # map subset indices back to global indices of the data base
                train_set = index_map[train_set]
                val_set = index_map[val_set]
                test_set = index_map[test_set]
        # store the set in which molecules are (1=train, 2=val, 3=test)
        train_val_test = np.zeros(len(db_validity), dtype=int)
        train_val_test[train_set] = 1
        train_val_test[val_set] = 2
        train_val_test[test_set] = 3
        # mark molecules that are not in a set as invalid
        # this excludes them from the comparison
        db_validity[train_val_test == 0] = 0
    else:
        train_val_test = None
    # compute lookup(s)
    db_unique, _, _, db_smiles_lookup = uniqueness_check_loop(
        validity=db_validity,
        smiles=db_smiles,
        formulas=db_formulas,
        ignore_enantiomers=ignore_enantiomers,
        ignore_isomerism=ignore_isomerism,
    )
    if unique is None or duplicating is None:
        unique, _, duplicating, _ = uniqueness_check_loop(
            validity=validity,
            smiles=smiles,
            formulas=formulas,
            ignore_enantiomers=ignore_enantiomers,
            ignore_isomerism=ignore_isomerism,
        )
    print(f"Running comparison...")
    known = np.ones(len(validity), dtype=int) * -1
    known_idx = np.ones(len(validity), dtype=int) * -1
    known_split = None
    if train_val_test is not None:
        known_split = np.ones(len(validity), dtype=int) * -1
    for i in tqdm(range(len(validity))):
        if validity[i]:
            if unique[i]:
                s = smiles[i]
                f = formulas[i]
                if ignore_isomerism:
                    s = Chem.CanonSmiles(s, useChiral=False)
                if f in db_smiles_lookup and s in db_smiles_lookup[f]:
                    known_idx[i] = db_smiles_lookup[f][s]
                    known[i] = 1
                    if train_val_test is not None:
                        known_split[i] = train_val_test[known_idx[i]]
                else:
                    known[i] = 0
                    if train_val_test is not None:
                        known_split[i] = 0
            else:
                d = duplicating[i]
                known_idx[i] = known_idx[d]
                known[i] = known[d]
                if train_val_test is not None:
                    known_split[i] = known_split[d]
    return known, known_idx, known_split


class SamplerWithSkipping(torch.utils.data.sampler.Sampler[int]):
    def __init__(self, dataset):
        self.dataset = dataset
        self.skip = 0

    def skip_steps(self, skip):
        self.skip = skip

    def __iter__(self):
        n = len(self.dataset)
        for i in range(self.skip, n):
            yield i

    def __len__(self):
        return len(self.dataset) - self.skip


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(
        data_path=args.data_path,
        results_path=args.results_path,
        allowed_charges=args.allowed_charges,
        allow_charged_fragments=args.allow_charged_fragments,
        allow_radical_electrons=args.allow_radical_electrons,
        compute_ring_statistics=args.compute_ring_statistics,
        force_recomputation=args.force_recomputation,
        ignore_enantiomers=args.ignore_enantiomers,
        ignore_isomerism=args.ignore_isomerism,
        compute_uniqueness=args.compute_uniqueness,
        compare_db_path=args.compare_db_path,
        compare_db_results_path=args.compare_db_results_path,
        compare_db_split_path=args.compare_db_split_path,
        timeout=args.timeout,
        write_to_db=args.write_to_db,
    )
