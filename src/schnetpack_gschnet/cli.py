import logging
import uuid
import tempfile
import shutil
import socket

import torch
import hydra
import numpy as np
import re
from omegaconf import DictConfig, OmegaConf, ListConfig, open_dict
from pathlib import Path
from ase.db import connect
from ase.data import chemical_symbols
from ase.visualize import view
from ase import Atoms
from tqdm import tqdm

from schnetpack.utils.script import print_config
from schnetpack_gschnet.generate_molecules import (
    generate_molecules,
    generate_molecules_debug,
)

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("uuid", lambda x: str(uuid.uuid1()))
OmegaConf.register_new_resolver("tmpdir", tempfile.mkdtemp, use_cache=True)


@hydra.main(config_path="configs", config_name="generate_molecules", version_base="1.2")
def generate(config: DictConfig):
    log.info(f"Running on host: {socket.gethostname()}")
    # create output file where the generated molecules will be written to
    outputdir = Path("./generated_molecules")
    if config.outputfile is not None:
        outputfile = (outputdir / config.outputfile).with_suffix(".db")
        outputfile.parent.mkdir(parents=True, exist_ok=True)
        outputfile.touch(exist_ok=True)
    else:
        count = 1
        while (outputdir / f"{count}.db").exists():
            count += 1
        outputfile = outputdir / f"{count}.db"
        outputfile.parent.mkdir(parents=True, exist_ok=True)
        outputfile.touch(exist_ok=False)

    # print config
    if config.get("print_config"):
        print_config(
            config,
            resolve=False,
            fields=("modeldir", "generate")
            if config.workdir is None
            else ("modeldir", "workdir", "remove_workdir", "generate"),
        )

    with connect(outputfile) as con:
        n_existing_mols = con.count()
        if n_existing_mols > 0:
            log.info(
                f"Caution, the data base {outputfile.resolve()} already exists and "
                f"contains {n_existing_mols} molecules. Generated molecules will be "
                f"appended to the data base."
            )

    original_outputfile = outputfile
    if config.workdir is not None:
        workdir = Path(config.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        outputfile = shutil.copy(original_outputfile, workdir)

    # choose device (gpu or cpu)
    if config.use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load model
    model = torch.load("best_model", map_location=device)

    # parse composition (if it is included in conditions)
    if "conditions" not in config.generate:
        with open_dict(config):
            config.generate.conditions = {}
    original_conditions = OmegaConf.to_container(
        config.settings.conditions.trajectory, resolve=True
    )
    config.settings.conditions = parse_composition(
        config.settings.conditions, model.get_available_atom_types().cpu().numpy()
    )

    # compute number of required batches
    n_batches = int(np.ceil(config.generate.n_molecules / config.generate.batch_size))
    if config.debug.run:
        n_batches = config.generate.n_molecules
        log.info(
            f"Caution: Using debug version of the generation function. The batch size "
            f"is automatically set to 1."
        )

    # generate molecules in batches
    log.info(
        f"Generating {config.generate.n_molecules} molecules in {n_batches} batches! "
        f'Running on device "{device}".'
    )
    ats = []
    with connect(outputfile) as con:
        for i in tqdm(range(n_batches)):
            # generate
            remaining = config.generate.n_molecules - i * config.generate.batch_size
            with torch.no_grad():
                if not config.debug.run:
                    R, Z, finished_list = generate_molecules(
                        model=model,
                        n_molecules=min(config.generate.batch_size, remaining),
                        max_n_atoms=config.generate.max_n_atoms,
                        grid_distance_min=config.generate.grid_distance_min,
                        grid_spacing=config.generate.grid_spacing,
                        conditions=config.settings.conditions,
                        device=device,
                        t=config.generate.temperature_term,
                        grid_batch_size=config.generate.grid_batch_size,
                    )
                else:
                    R, Z, finished_list = generate_molecules_debug(
                        model=model,
                        max_n_atoms=config.generate.max_n_atoms,
                        grid_distance_min=config.generate.grid_distance_min,
                        grid_spacing=config.generate.grid_spacing,
                        conditions=config.settings.conditions,
                        device=device,
                        t=config.generate.temperature_term,
                        print_progress=config.debug.print_progress,
                        view_progress=config.debug.view_progress,
                    )

                # store generated molecules in db
                R = R.cpu().numpy()
                Z = Z.cpu().numpy()
                for idx, n_atoms in finished_list:
                    at = Atoms(
                        numbers=Z[idx, 2 : n_atoms + 2],
                        positions=R[idx, 2 : n_atoms + 2],
                    )
                    ats += [at]
                    con.write(at)

        # update metadata of the data base
        n_new_mols = con.count() - n_existing_mols
        key = f"{n_existing_mols}-{n_existing_mols + n_new_mols - 1}"
        md = con.metadata
        md.update(
            {
                key: {
                    "conditions": original_conditions,
                    "atom_types": model.get_available_atom_types()
                    .cpu()
                    .numpy()
                    .tolist(),
                }
            }
        )
        con.metadata = md
    if config.workdir is not None:
        shutil.copy(outputfile, original_outputfile)
        if config.remove_workdir:
            shutil.rmtree(workdir)

    log.info(
        f"Finished generation. Wrote {n_new_mols} successfully generated molecules to "
        f"the data base at {original_outputfile.resolve()}!"
    )

    # visualize molecules if desired
    if config.view_molecules:
        view(ats)


def parse_composition(conditions, available_atom_types):
    # check if `composition` is in conditions
    if (
        conditions is None
        or "trajectory" not in conditions
        or conditions["trajectory"] is None
        or "composition" not in conditions["trajectory"]
    ):
        return conditions
    else:
        composition = conditions.trajectory.composition
    atom_names = [chemical_symbols[atom_type] for atom_type in available_atom_types]
    composition_list = [0 for _ in range(len(available_atom_types))]
    composition_dict = {name: 0 for name in atom_names}
    # composition is given as list
    if isinstance(composition, ListConfig):
        if isinstance(composition[0], int):
            # list of integers (assumed to have the same order as in `atom_names`)
            composition_list = composition
            for atom_name, number in zip(atom_names, composition):
                composition_dict[atom_name] = number
        elif isinstance(composition[0], ListConfig):
            # list of lists (assumes that there are 2 entries per list, element+number)
            for atom_name, number in composition:
                if isinstance(atom_name, int):
                    atom_name = chemical_symbols[atom_name]
                composition_dict[atom_name.title()] = number
                composition_list[atom_names.index(atom_name.title())] = number
        elif isinstance(composition[0], DictConfig):
            # list of dictionaries that have elements as keys and numbers as values
            for entry in composition:
                for atom_name in entry:
                    number = entry[atom_name]
                    if isinstance(atom_name, int):
                        atom_name = chemical_symbols[atom_name]
                    composition_dict[atom_name.title()] = number
                    composition_list[atom_names.index(atom_name.title())] = number
    # composition is given as dictionary with elements as keys and numbers as values
    elif isinstance(composition, DictConfig):
        for key in composition:
            if isinstance(key, int):
                # key is the atomic number -> translate to name
                atom_name = chemical_symbols[key]
            else:
                atom_name = key.title()
            composition_dict[atom_name] = composition[key]
            composition_list[atom_names.index(atom_name)] = composition[key]
    # composition is given as string (e.g. `C7O2H10`)
    elif isinstance(composition, str):
        for atom_name, number in re.findall(r"(\w+?)(\d+)", composition):
            composition_dict[atom_name.title()] = int(number)
            composition_list[atom_names.index(atom_name.title())] = int(number)
    else:
        raise ValueError(
            f"Composition provided in wrong format! Please provide a molecular "
            f"formula (e.g. C6O1H10), a dictionary with chemical elements as keys and "
            f"the number of atoms of that element as values, or a list of the number "
            f"of atoms of each element in the order {atom_names}."
        )
    composition_str = ""
    H_str = ""
    for atom_name in composition_dict:
        if composition_dict[atom_name] > 0:
            if atom_name != "H":
                composition_str += f"{atom_name}{composition_dict[atom_name]}"
            else:
                H_str += f"{atom_name}{composition_dict[atom_name]}"
    conditions.trajectory.composition = ListConfig(composition_list)
    return conditions
