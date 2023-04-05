# Conditional G-SchNet extension for SchNetPack 2.0 - A generative neural network for 3d molecules

![generated molecules](https://github.com/atomistic-machine-learning/G-SchNet/blob/master/images/example_molecules_1.png)

[G-SchNet](http://papers.nips.cc/paper/8974-symmetry-adapted-generation-of-3d-point-sets-for-the-targeted-discovery-of-molecules) is a generative neural network that samples molecules by sequentially placing atoms in 3d space.
It can be trained on data sets of 3d molecules with variable sizes and compositions.
The conditional version, [cG-SchNet](https://www.nature.com/articles/s41467-022-28526-y), explicitly takes chemical and structural properties into account to allow for targeted molecule generation.

Here we provide a re-implementation of [cG-SchNet](https://github.com/atomistic-machine-learning/G-SchNet) using the updated [SchNetPack 2.0](https://github.com/atomistic-machine-learning/schnetpack/tree/master).
Compared to previous releases, SchNetPack changed from batching molecules to batching atoms, effectively removing the need for padding the neural network inputs.
G-SchNet greatly benefits from this change in terms of memory requirements, allowing to train models of the same expressivity on GPUs with less VRAM.

Furthermore, we [altered a few implementation details](/README.md#changes-in-this-implementation) to improve scalability and simplify adaptations to custom data sets. 
Therefore, we recommend this version for applications of G-SchNet to new data sets and further development of the method.
For reproduction of the results reported in our publications, please refer to the specific repositories:
-  [G-SchNet](https://github.com/atomistic-machine-learning/G-SchNet) ([Symmetry-adapted generation of 3d point sets for the targeted discovery of molecules, 2019](http://papers.nips.cc/paper/8974-symmetry-adapted-generation-of-3d-point-sets-for-the-targeted-discovery-of-molecules))
- [cG-SchNet](https://github.com/atomistic-machine-learning/cG-SchNet) ([Inverse design of 3d molecular structures with conditional generative neural networks, 2022](https://www.nature.com/articles/s41467-022-28526-y))

### Content

+ [Installation](/README.md#installation)
+ [Command-line interface and configuration](/README.md#command-line-interface-and-configuration)
  + [Model training](/README.md#model-training)
    + [Hyperparameters and experiment settings](/README.md#hyperparameters-and-experiment-settings)
    + [Specifying target properties](/README.md#specifying-target-properties)
    + [Using custom data](/README.md#using-custom-data)
    + [Scaling up the training](/README.md#scaling-up-the-training)
  + [Molecule generation](/README.md#molecule-generation)
+ [Additional information](/README.md#additional-information)
  + [FAQ and troubleshooting](/README.md#faq-and-troubleshooting)
  + [Changes in this implementation](/README.md#changes-in-this-implementation)
  + [Citation](/README.md#citation)
  + [How does cG-SchNet work?](/README.md#how-does-cg-schnet-work)

# Installation

To install `schnetpack-gschnet`, download this repository and use pip.
For example, the following commands will clone the repository into your current working directory, create a new conda environment called `gschnet`, and install this package as well as all its dependencies (e.g. SchNetPack 2.0, PyTorch, etc.) in the new environment (tested on Ubuntu 20.04):

```
git clone https://github.com/atomistic-machine-learning/schnetpack-gschnet.git
conda create -n gschnet numpy
conda activate gschnet
pip install ./schnetpack-gschnet
```

# Command-line interface and configuration

The `schnetpack-gschnet` package is built on top of `schnetpack` [version 2.0](https://github.com/atomistic-machine-learning/schnetpack/tree/master), which is a library for atomistic neural networks with flexible customization and configuration of models and experiments.
It is integrated with the [PyTorch Lightning](https://www.pytorchlightning.ai/) learning framework, which takes care of the boilerplate code required for training and provides a standardized, modular interface for incorporating learning tasks and data sets.
Moreover, SchNetPack utilizes the hierarchical configuration framework [Hydra](https://hydra.cc/).
This allows to define training runs using YAML config files that can be loaded, composed, and overridden via a powerful command-line interface (CLI).

`schnetpack-gschnet` is designed to leverage both the PyTorch Lightning integration and the hierarchical Hydra config files.
The [configs directory](/src/schnetpack_gschnet/configs) contains the YAML files that specify different set-ups for training and generation.
It exactly follows the structure of the [configs directory](https://github.com/atomistic-machine-learning/schnetpack/src/schnetpack/configs) from `schnetpack`.
In this way, we can compose a config using files from `schnetpack`, e.g. for the optimizer, as well as new config files from `schnetpack-gschnet`, e.g. for the generative model and the data.
We recommend to copy the configs directory from `schnetpack-gschnet` to create a personal resource of config files, e.g. with:

```
cp -r <path/to/schnetpack-gschnet>/src/schnetpack_gschnet/configs/. <path/to/my_gschnet_configs>
```

You can customize the existing configs or create new ones in that directory, e.g. to set up training of cG-SchNet with custom conditions.
All hyperparameters specified in the YAML files can also directly be set in the CLI when calling the training or generation script.
We will explain the most important hyperparameters and config files in the following sections on training and molecule generation.
For more details on the strucure of the config, the CLI, and the PyTorch Lightning integration, please refer to the [software paper](https://arxiv.org/abs/2212.05517) for SchNetPack 2.0 and the [documentation](https://schnetpack.readthedocs.io/en/latest/) of the package.

## Model training

If you have copied the configs directory as recommended [above](/README.md#configuration-and-cli), the following call will start a training run in the current working directory:

```
python <path/to/schnetpack-gschnet>/src/scripts/train.py --config-dir=<path/to/my_gschnet_configs> experiment=gschnet_qm9
```

The call to the training script requires two arguments, the directory with configs from `schnetpack-gschnet` and the name of the experiment config you want to run.
The experiment config is the most important file for the configuration of the training run.
It determines all the hyperparameters, i.e. which model, dataset, optimizer, callbacks etc. to use.
We provide three examplary [experiment configs](/src/schnetpack_gschnet/configs/experiment):

| Experiment name | Description |
| :--- | :--- |
|`gschnet_qm9`| Trains an unconditioned G-SchNet model on the QM9 data set. Mostly follows the experimental setup described in the sections 5, 5.1, and 5.2 of the [G-SchNet publication](https://proceedings.neurips.cc/paper/2019/file/a4d8e2a7e0d0c102339f97716d2fdfb6-Paper.pdf). |
|`gschnet_qm9_comp_relenergy`| Trains a cG-SchNet model that is conditioned on the atomic composition and the relative atomic energy of molecules on the QM9 data set. Mostly follows the experimental setup described in the section "Discovery of low-energy conformations" of the [cG-SchNet publication](https://www.nature.com/articles/s41467-022-28526-y). Accordingly, a filter is applied to exclude all C<sub>7</sub>O<sub>2</sub>H<sub>10</sub> conformations from the training and validation splits. |
|`gschnet_qm9_gap_relenergy`| Trains a cG-SchNet model that is conditioned on the HOMO-LUMO gap and the relative atomic energy of molecules on the QM9 data set. Mostly follows the experimental setup described in the section "Targeting multiple properties: Discovery of low-energy structures with small HOMO-LUMO gap" of the [cG-SchNet publication](https://www.nature.com/articles/s41467-022-28526-y). |

Simply change the experiment name in the call to the training script to run any of the three example experiments.
PyTorch Lightning will automatically use a GPU for training if it can find one, otherwise the training will run on CPU.
Please note that we do not recommend training on the CPU as it will be very slow.
In the following sections, we explain the most important settings and how to customize the training run, e.g. to change the target properties for conditioning of the model or to use a custom data set instead of QM9.

### Hyperparameters and experiment settings

The three provided experiment configs mostly use the same hyperparameters and only differ in the properties the generative model is conditioned on.
In the following table we list the most important hyperparameters and settings including their default value in the three experiments.
Here, `${}` invokes variable interpolation, which means that the value of another hyperparameter from the config is inserted or a special resolver is used, e.g. to get the current working directory.
All these settings can easily be changed in the CLI, e.g. using `trainer.accelerator=gpu` as additional argument to make sure that training runs on the gpu.

| Name | Value | Description |
| :--- | :--- | :--- |
| `run.work_dir` | `${hydra:runtime.cwd}` | The root directory for running the script. The default value sets it to the current working directory. |
| `run.data_dir` | `${run.work_dir}/`<br>`data` | The directory where training data is stored (or will be downloaded to). |
| `run.path` | `${run.work_dir}/`<br>`models/`<br>`qm9_${globals.name}` | The path where the directory with results of the run (e.g. the checkpoints, the used config etc.) will be stored. For our three experiments, `globals.name` is set to the properties the model is conditioned on, i.e. _no\_conditions_, _comp\_relenergy_, and _gap\_relenergy_, respectively. |
| `run.id` | `${globals.id}` | The name of the directory where the results of the run will be stored. The default setting of `globals.id` automatically generates a unique identifier or, when available, uses the job id and hostname. |
| `trainer.accelerator` | `auto` | The type of the accelerator used for training. Supports `cpu`, `gpu`, `tpu`, `ipu`, and `auto`. The `auto` option tries to select the best accelerator automatically. Please beware that training on the CPU is not recommended as it is very slow and that we did not test this package with accelerators other than `gpu`. |
| `globals.lr` | `1e-4` | The learning rate for the optimizer at the start of the training. We use the Adam optimizer and a reduce on plateau learning rate scheduler, the corresponding settings can be found in the [gschnet_task config](/src/schnetpack_gschnet/configs/task/gschnet_task.yaml). |
| `globals.atom_types` | `[1, 6, 7, 8, 9]` | List of all the atom types contained in molecules of the training data set (expressed as nuclear charges, i.e. here we have H, C, N, O, and F). |
| `globals.origin_type` | `121` | Type used for the auxiliary origin token (cannot be contained in `globals.atom_types`). The origin token marks the center of mass of training structures and is the starting point for the trajectories of atom placements, i.e. molecules grow around the origin token. |
| `globals.focus_type` | `122` | Type used for the auxiliary focus token (cannot be contained in `globals.atom_types`). At each step, the focus is aligned with a randomly selected atom and the atom placed next needs to be a neighbor of that focused atom. |
| `globals.stop_type` | `123` | Type used for the stop marker that G-SchNet predicts to mark that it cannot place more atoms in the neighborhood of the current focus (cannot be contained in `globals.atom_types`). |
| `globals.model_cutoff` | `10.` | The cutoff used in the interaction blocks of the SchNet model which extracts features from the intermediate molecular structures. |
| `globals.prediction_cutoff` | `10.` | The cutoff used to determine for which atoms around the focus the distance to the new atom is predicted. |
| `globals.placement_cutoff` | `1.7` | The cutoff used to determine which atoms can be placed (i.e. which are neighbors of the focus) when building a trajectory of atom placements for training. |
| `globals.use_covalent_radii` | `True` | If True, the covalent radii of atom types are additionally used to check whether atoms that are inside the `globals.placement_cutoff` are neighbors of the focus. We use the covalent radii provided in the `ase` package and check whether the distance between the focus and another atom is smaller than the sum of the covalent radii for the types of the two atoms scaled by `globals.covalent_radii_factor`. |
| `globals.covalent_radius_factor` | `1.1` | Scales the sum of the two covalent radius numbers to relax the neighborhood criterion when `globas.use_covalent_radii` is True. |
| `globals.draw_random_samples` | `0` | The number of atom placements that are randomly drawn per molecule in the batch in each epoch (without replacement, i.e. each step can at most occure once). If `0`, all atom placements are used, which means that the number of atoms in a batch scales quadratically with the number of atoms in the training molecules. Therefore, we recommend to select a value larger than `0` to have linear scaling when using data sets with molecules larger than those in QM9. |
| `globals.data_workdir` | `null` | Path to a directory where the data is copied to for fast access (e.g. local storage of a node when working on a cluster). If null, the data is loaded from its original destination at `data.datapath` each epoch. |
| `globals.cache_workdir` | `null` | Path to a directory where the data cache is copied to for fast access (e.g. local storage of a node when working on a cluster). Only used if the results of the neigborhood list are cached. If null, the cached data is loaded from its original destinaltion each epoch. |
| `data.batch_size` | `5` | The number of molecules in training and validation batches. Note that each molecule occurs multiple times in the batch as they are reconstructed in a trajectory of atom placements. The number of times they occur can be limited with `globals.draw_random_samples`. |
| `data.num_train` | `50000` | The number of molecules for the training split of the data. |
| `data.num_val` | `5000` | The number of molecules for the validation split of the data. |
| `data.datapath` | `${run.data_dir}/`<br>`qm9.db` | The path to the training data base file. |
| `data.remove_uncharacterized` | `True` | Whether the molecules marked as uncharacterized in QM9 are removed from the training data base. Note that the data base has to be re-downloaded and built if this setting is changed. |
| `data.num_workers` | `6` | The number of CPU workers spawned to load training data batches. |
| `data.num_val_workers` | `4` | The number of CPU workers spawned to load validation data batchs. |
| `data.num_test_workers` | `4` | The number of CPU workers spawned to load test data batches.  |
| `data.distance_unit` | `Ang` | The desired distance unit used for the coordinates of atoms. The conversion is automatically done by SchNetPack if the distance unit in the data base is different. |
| `data.property_units.energy_U0` | `eV` | The desired unit of the property _energy\_U0_. The conversion is automatically done by SchNetPack if the unit in the data base is different. |
| `data.property_units.gap` | `eV` | The desired unit of the property _gap_. The conversion is automatically done by SchNetPack if the unit in the data base is different.  |
| `callbacks.early_stopping.`<br>`patience` | `25` | The number of epochs after which the training is stopped if the validation loss did not improve. On QM9, cG-SchNet typically trains for 150-250 epochs with these settings. |
| `callbacks.progress_bar.`<br>`refresh_rate` | `100` | The number of batches processed before the progress bar is refreshed. |

### Specifying target properties

In cG-SchNet, the target properties for the conditional distribution are embedded with a neural network block.
This is implemented in the class [ConditioningModule](https://github.com/atomistic-machine-learning/schnetpack-gschnet/blob/724ddc6c1ed965e0900e74f197225fe7bbcadd51/src/schnetpack_gschnet/model.py#L334).
First, each property is embedded with an individual network and then all embeddings are concatenated and processed through another block of fully connected layers.
In this way, we can use any combination of target properties as conditions as long as we have embedding networks for each individual property.
The base class for the embedding networks is [ConditionEmbedding](https://github.com/atomistic-machine-learning/schnetpack-gschnet/blob/724ddc6c1ed965e0900e74f197225fe7bbcadd51/src/schnetpack_gschnet/model.py#L398) and the package contains three subclasses:
[ScalarConditionEmbedding](https://github.com/atomistic-machine-learning/schnetpack-gschnet/blob/724ddc6c1ed965e0900e74f197225fe7bbcadd51/src/schnetpack_gschnet/model.py#L451) for scalar-valued properties (e.g. energy, HOMO-LUMO gap, etc.),
[VectorialConditionEmbedding](https://github.com/atomistic-machine-learning/schnetpack-gschnet/blob/724ddc6c1ed965e0900e74f197225fe7bbcadd51/src/schnetpack_gschnet/model.py#L526) for vector-valued properties (e.g. fingerprints), and
[CompositionEmbedding](https://github.com/atomistic-machine-learning/schnetpack-gschnet/blob/724ddc6c1ed965e0900e74f197225fe7bbcadd51/src/schnetpack_gschnet/model.py#L587) for the composition of molecules.

To specify a set of target properties for an experiment, we set up the corresponding `ConditioningModule` in a config file.
For example, the experiment `gschnet_qm9_gap_relenergy`, which targets HOMO-LUMO gap and relative atomic energy as conditions, uses the following config:

https://github.com/atomistic-machine-learning/schnetpack-gschnet/blob/724ddc6c1ed965e0900e74f197225fe7bbcadd51/src/schnetpack_gschnet/configs/model/conditioning/gap_relenergy.yaml#L1-L22

Both, the energy and the gap, are embedded using a `ScalarConditionEmbedding`.
It projects a scalar value into vector space using a Gaussian expansion with centers between `condition_min` and `condition_max` and a spacing of `grid_spacing`, i.e. 5 centers in the examples above.
Then, it applies a network consisting of three fully connected layers with 64 neurons to extract a vector with 64 features for the corresponding property.
The individual vectors of both porperties are concatenated and then processed by five fully connected layers with 128 neurons to obtain a final vector with 128 features that jointly represents energy and gap.
Please note that the values for `condition_min`, `condition_max`, and `grid_spacing` of the Gaussian expansion need to be expressed in the units used for the properties.
For example, we use _eV_ in this case for both gap and energy as specified in the default [data config](/src/schnetpack_gschnet/configs/data/gschnet_qm9.yaml) for the QM9 data set.

An important argument in the config is `required_data_properties`.
It determines additional properties that need to be loaded from the data base for the conditioning.
Here, the relative atomic energy is a special case as it is not directly included in the QM9 data base.
Instead, we load the total energy at zero Kelvin `energy_U0` and compute the relative atomic energy from this.
To this end, there exist `transforms` that are applied to every data point by the data loader.
The `transforms` are specified in the experiment config as part of the `data` field:

https://github.com/atomistic-machine-learning/schnetpack-gschnet/blob/724ddc6c1ed965e0900e74f197225fe7bbcadd51/src/schnetpack_gschnet/configs/experiment/gschnet_qm9_gap_relenergy.yaml#L44-L69

The corresponding `GetRelativeAtomicEnergy` transform is defined in lines 51-54.
On a side note, we see that transforms take care of all kind of preprocessing tasks, e.g. centering the atom positions, computing neighborhood lists of atoms, and sampling a trajectory of atom placements for the molecule.

As another example, you can refer to the experiment `gschnet_qm9_comp_relenergy`, which targets the atomic composition and the relative atomic energy as conditions.
There, the [conditioning config](/src/schnetpack_gschnet/configs/model/conditioning/comp_relenergy.yaml) uses the same `ScalarConditionEmbedding` as before for the relative atomic energy but combines it with a `CompositionEmbedding` for the atomic composition.
Accordingly, the `transforms` in the [experiment config](/src/schnetpack_gschnet/configs/experiment/gschnet_qm9_comp_relenergy.yaml) contain `GetRelativeAtomicEnergy` and `GetComposition` to compute the relative atomic energy and the atomic composition, respectively.

To summarize, we can specify a set of target properties by adding a conditioning config to `<path/to/my_gschnet_configs>/model/conditioning`.
If the target properties can directly be loaded from the data base, we can use the basic `gschnet_qm9` experiment and append our new conditioning config in the CLI to start training.
For example, suppose we want to only condition our model on the HOMO-LUMO gap.
To this end, we can delete lines 14-22 in the `gap_relenergy` conditioning config shown above and save the resulting file as `<path/to/my_gschnet_configs>/model/conditioning/gap.yaml`.
Then, the training can be started with:

```
python <path/to/schnetpack-gschnet>/src/scripts/train.py --config-dir=<path/to/my_gschnet_configs> experiment=gschnet_qm9 model/conditioning=gap
```

If the target properties are not stored in the data base, it is most convenient to set up an experiment config with suitable `transforms` that compute them.
We directly link the conditioning config in the experiment config by overriding `/model/conditioning` in the defaults list, as can be seen in the last line of the following example:

https://github.com/atomistic-machine-learning/schnetpack-gschnet/blob/724ddc6c1ed965e0900e74f197225fe7bbcadd51/src/schnetpack_gschnet/configs/experiment/gschnet_qm9_gap_relenergy.yaml#L3-L13

Then, we only provide the name of the new experemint config in the CLI and do not need an additional argument for the conditioning config.

### Using custom data

In order to use custom data, we need to store it in the [ASE data base format](https://wiki.fysik.dtu.dk/ase/ase/db/db.html) that is used in `schnetpack`.
The preparation of such a data base takes only a few steps and can be done with the help of `schnetpack.data.ASEAtomsData`.
For example, if you have a function `read_molecule` that gives the atom positions, atom types, and property values of a molecule (e.g. from _xyz_, _cif_, or another _db_ file), you can create a data base in the correct format with the following code:

```python
from ase import Atoms
from schnetpack.data import ASEAtomsData
import numpy as np

mol_list = []
property_list = []
for i in range(n_molecules):
    # get molecule information with your custom read_molecule function
    atom_positions, atom_types, property_values = read_molecule(i)
    # create ase.Atoms object and append it to the list
    mol = Atoms(positions=atom_positions, numbers=atom_types)
    mol_list.append(mol)
    # create dictionary that maps property names to property values and append it to the list
    # note that the property values need to be numpy float arrays (even if they are scalar values)
    properties = {
        "energy": np.array([float(property_values[0])]), 
        "gap": np.array([float(property_values[1])]),
    }
    property_list.append(properties)

# create empty data base with correct format
# make sure to provide the correct units of the positions and properties
custom_dataset = ASEAtomsData.create(
    "/home/user/custom_dataset.db",                              # where to store the data base
    distance_unit="Angstrom",                                    # unit of positions
    property_unit_dict={"energy": "Hartree", "gap": "Hartree"},  # units of properties
)
# write gathered molecules and their properties to the data base
custom_dataset.add_systems(property_list, mol_list)
```

In the for-loop, we build a list of molecular structures in the form of `ase.Atoms` objects and a corresponding list of dictionaries containing mappings from property names to property values for each molecule.
Having these lists, we can easily create an empty data base in the correct format and store our gathered molecules with functions from `schnetpack.data.ASEAtomsData`.
In this example, we assume that we have _energy_ and _gap_ values in _Hartree_ for each molecule and that the atom positions are give in _Angstrom_.
Of course, the `read_molecule` function and lines where we specify properties and units need to be adapted carefully to fit your custom data if you use the code from above.
To find which units are supported, please check the [ASE units module](https://wiki.fysik.dtu.dk/ase/ase/units.html).
It includes most common units such as `eV`, `Ha`, `Bohr`, `kJ`, `kcal`, `mol`, `Debye`, `Ang`, `nm` etc.

Once the data is in the required format, we can train G-SchNet.
To this end, we provide the `gschnet_template` experiment config and the `template` data config:

https://github.com/atomistic-machine-learning/schnetpack-gschnet/blob/d6e2fd13a46cb04d1617d347f8802fc0f385835a/src/schnetpack_gschnet/configs/experiment/gschnet_template.yaml#L1-L62
https://github.com/atomistic-machine-learning/schnetpack-gschnet/blob/d6e2fd13a46cb04d1617d347f8802fc0f385835a/src/schnetpack_gschnet/configs/data/template.yaml#L1-L14

Here, arguments specific to the custom data set are left with `???`, wich means that they need to be specified in the CLI when using the configs.
For example, assume we have stored a data base with 10k molecules consisting of carbon, oxygen, and hyrogen at _/home/user/custom_dataset.db_.
Then, we can start the training process with the following, long call:

```
python <path/to/schnetpack-gschnet>/src/scripts/train.py --config-dir=<path/to/my_gschnet_configs> experiment=gschnet_template data.datapath=/home/user/custom_dataset.db data.batch_size=10 data.num_train=5000 data.num_val=2000 globals.name=custom_data globals.id=first_try globals.model_cutoff=10 globals.prediction_cutoff=10 globals.placement_cutoff=1.7 globals.atom_types="[1, 6, 8]"
```

Alternatively, you can copy the configs and fill in the left-out arguments in the files.
Please choose them according to your data, e.g the `placement_cutoff` should be slightly larger than the typical bond lengths.
Systems with periodic boundary conditions are currently not supported.
By default, a model without target properties is trained.
To train a model with conditions, you need to create a conditioning config as explained in the [previous section](/README.md#specifying-target-properties).
To convert the units of target properties upon loading, add `property_units` to the data config (cf. the [QM9 data config](/src/schnetpack_gschnet/configs/data/gschnet_qm9.yaml)).
Also, note our hints on scaling up the training in the following section if your data set includes large molecules.

### Scaling up the training

The QM9 data set, which is used in the example experiments, contains only small organic compounds.
Therefore, the default settings in those configs might lead to excessive memory and runtime requirements when using other data sets.
In the following, we shed light on the most important settings and tweaks to use G-SchNet with molecules much larger than those in QM9.

#### 1. Set draw_random_samples > 0

In each epoch, the data loader reads each molecule from the data base and samples a trajectory of atom placements, i.e. it starts with the first atom, then selects a second atom, a third atom, and so on.
When predicting the position of the second atom, the model uses the structure consisting of only the first atom, when predicting the position of the third atom, the model uses the structure consisting of the two first atoms etc.
Therefore, the partial molecule occurs many times in a batch, once for every prediction step.
Accordingly, the number of atoms in a batch scales quadratically with the number of atoms in the training structures.
For QM9 this is feasible but it becomes problematic when working with larger structures.
To remedy this effect, we can restrict the batch to contain only a fixed number of prediction steps per molecule, which confines it to scale linearly.
The config setting `globals.draw_random_samples` determines the number of steps that are drawn randomly from the whole trajectory.
For data sets with large molecules, we recommend to set it to a small number, e.g. five.
Setting it to zero restores the default behavior, which adds all steps to the batch.
Please note that the prediction steps for the validation batch will also be drawn randomly, which causes the validation loss to vary slightly even if the model does not change.

#### 2. Choose suitable cutoffs

Two other parameters that influence the scaling of the method are the prediction cutoff and the model cutoff.
The network will only predict the distances of the new atom to atoms that are closer to the focus than the prediction cutoff.
This limits the number of distributions predicted at each step and therefore improves the scaling compared to previous implementations, where the distances to all preceding atoms were predicted.
Choosing a larg prediction cutoff will lead to higher flexibility of the network but also higher memory consumption and potentially redundant predictions.
A very small prediction cutoff might hurt the performance of the model.
The model cutoff determines the neighbors that exchange messages when extracting atom-wise features from the molecular structure.
A large model cutoff can be interpreted as increasing the receptive field of an atom but comes with higher computational costs.
From our experience, we recommend to set both cutoffs to similar values.
The corresponding config settings are `globals.prediction_cutoff` and `globals.model_cutoff`.
For QM9, we use 10 Angstrom, which should also be a reasonable starting point for other data sets.

#### 3. Use caching of neighborlists

The data loading and preprocessing can become quite expensive for larger structures, especially the computation of the neighborlists of all atoms.
The batches are loaded in parallel to neural network computations by separate CPU threads.
If the GPU has a low utilization during training, it might help to use more workers to reduce the waiting time of the GPU.
The number of workers for training, validation, and test data can be set with `data.num_workers`, `data.num_val_workers`, and `data.num_test_workers`, respectively.
However, for larger molecules we generally recommend to cache the computed neighborlists to reduce the load of the workers.
To this end, the package contains the `GeneralCachedNeighborList` [transform](https://github.com/atomistic-machine-learning/schnetpack-gschnet/blob/724ddc6c1ed965e0900e74f197225fe7bbcadd51/src/schnetpack_gschnet/transform/neighborlist.py#L33).
It can be incorporated in the experiment config by wrapping `ConditionalGSchNetNeighborList` in the list of transforms as follows:

```yaml
    - _target_: schnetpack_gschnet.transform.GeneralCachedNeighborList
      cache_path: ${run.work_dir}/cache
      keep_cache: False
      cache_workdir: ${globals.cache_workdir}
      neighbor_list:
        _target_: schnetpack_gschnet.transform.ConditionalGSchNetNeighborList
        model_cutoff: ${globals.model_cutoff}
        prediction_cutoff: ${globals.prediction_cutoff}
        placement_cutoff: ${globals.placement_cutoff}
        environment_provider: ase
        use_covalent_radii: ${globals.use_covalent_radii}
        covalent_radius_factor: ${globals.covalent_radius_factor}
```

The cache will be stored at `cache_path` and deleted after training unless `keep_cache` is set to `True`.
As the neighborlist results depend on the chosen cutoffs, do not re-use the cache from previous runs unless you are 100% sure that all settings are identical.

#### 4. Use working directories for data and caching

Another reason for low GPU utilization can be slow reading speed of the data and cache.
For example, when running code on a cluster, you often have slow, shared storage and faster, local node storage.
Therefore, you can choose working directories for the data and cache by setting `globals.data_workdir` and `globals.cache_workdir` to directories on the fast storage.
If you do so, the data and cache will be copied to these locations and then read from there for the training run.
They are automatically deleted if the run finishes without errors.

## Molecule generation

After training a model, you can generate molecules from the CLI with the generation script:

```
python <path/to/schnetpack-gschnet>/src/scripts/generate.py --config-dir=<path/to/my_gschnet_configs> modeldir=<path/to/trained/model>
```

The call to the generation script requires two arguments, the directory with configs from `schnetpack-gschnet` and the path to the root directory of the trained model, i.e. the directory containing the files _best\_model_, _cli.log_, _config.yaml_ etc.
The generated molecules are stored in an `ASE` data base at `<modeldir>/generated_molecules/`.
For models trained with conditions, target values for all properties that were used have to be specified.
For example, for a model trained with the `gschnet_qm9_gap_relenergy` config, both a target HOMO-LUMO gap and relative atomic energy have to be set.
This can be done by appending the following arguments to the CLI call:

```
+generate.conditions.gap=4.0 +generate.conditions.relative_atomic_energy=-0.2
```

Here the `+` is needed to append new arguments to the config (as opposed to setting new values for existing config entries).
Alternatively, you can create a config file for the target property values at `<path/to/my_gschnet_configs>/generate/conditions/my_conditions.yaml` and append it to the config by adding `generate/conditions=my_conditions` to the CLI call.
The package contains two [exemplary target value config files](/src/schnetpack_gschnet/configs/generate/conditions) that cover the two cG-SchNet example experiments using the target values from the corresponding experiments in the publication.
Note that the names of the target properties have to correspond to the `condition_name` specified in the conditioning configs of the trained model.
That is why we use _gap_ and _relative\_atomic\_energy_ here, as specified in lines 6 and 15 of the [conditioning config](/src/schnetpack_gschnet/configs/model/conditioning/gap_relenergy.yaml).
In models conditioned on the atomic composition, the corresponding property name is automatically set to _composition_.

In the following table, we list all settings for the generation script and their default values.
All settings can directly be set in the CLI, e.g. add `generate.n_molecules=1000` to the call to generate a thousand instead of a hundred molecules.

| Name | Value | Description |
| :--- | :--- | :--- |
| `generate.n_molecules` | `100` | The number of molecules that shall be generated. Note that the number of molecules in the resulting data base can be lower as failed generation attempts are not stored, i.e. where the model has not finished generation after placing `max_n_atoms` atoms. |
| `generate.batch_size` | `10` | The number of molecules generated in one batch. Use large batches if possible and decrease the batch size if your GPU runs out of memory. |
| `generate.max_n_atoms` | `35` | The maximum number of atoms the model is allowed to place. If it has not finished after placing this many atoms, it will discard the structure as a failed generation attempt. |
| `generate.grid_distance_min` | `0.7` | The minimum distance between a new atom and the focus atom. Determines the extent of the 3d grid together with the `placement_cutoff` used during model training, which sets the maximum distance between new atom and focus. |
| `generate.grid_spacing` | `0.05` | The size of a bin in the 3d grid (i.e. a value of 0.05 means each bin has a size of 0.05x0.05x0.05). |
| `generate.temperature_term` | `0.1` | The temperature term in the normalization of the 3d grid probability. A smaller value leads to more pronounced peaks whereas a larger value increases randomness by smoothing the distribution. |
| `generate.grid_batch_size` | `0` | For the reconstruction of the positional distributions, one 3d grid is constructed for every preceding atom (within the `prediction cutoff` of the model). This operation consumes a lot of memory and therefore is a bottleneck when it comes to the number of molecules that can be generated at the same time. When setting this argument to an integer `x > 0`, at most `x` 3d grids are constructed at the same time, which allows to control the memory demand of the generation process. The default value of `0` means that all grids are computed at once. |
| `outputfile` | `null` | Name of the data base where generated molecules are stored. The data base will always be stored at `<path/to/trained/model>/generated_molecules/`. If `null`, the script will automatically assign a number to the data base (it starts to count from 1 and increases the count by one if a data base with the number already exists). |
| `use_gpu` | `True` | Set `True` to run generation on the GPU. |
| `view_molecules` | `False` | Set `True` to automatically open a pop-up window with visualizations of all generated structures (uses the `ASE` package for visualization). |
| `workdir` | `null` | Path to a directory. If not `null`, the data base will first be written to this directory and then copied to `<path/to/trained/model>/generated_molecules/`. This can speed up the generation if the storage of `<path/to/trained/model>` is slow, e.g. a shared drive on a cluster, and `workdir` is on a fast local storage. If the directory does not exist, it will be created automatically. |
| `remove_workdir` | `False` | If `True`, the `workdir` is automatically removed after the data base has been copied to `<path/to/trained/model>/generated_molecules`. |

# Additional information

## FAQ and troubleshooting

### 1. Can I restart training after a run crashed or timed out?
Yes, if you start a run with an existing `run.id`, the training will automatically be resumed from the last stored checkpoint.
The `run.id` is the name of the folder where the model, logs, config etc. of a run are stored.
In our example configs, `run.id` is automatically set to a unique identifier if you do not provide it manually.
Accordingly, every time a run is started, a new folder is created and a new model is trained.
However, if you set `run.id=<your_name>` in your CLI call the training will be resumed if there already exists a folder called `<your_name>` (otherwise, a fresh run with that name will be initialized). 

### 2. Data preprocessing is getting stuck or crashing
We use multi-processing to speed up the data setup phase before training starts.
On some machines, this can lead to [the process getting stuck](https://github.com/atomistic-machine-learning/schnetpack-gschnet/issues/3) or [crashes with error messages related to pickle](https://github.com/atomistic-machine-learning/cG-SchNet/issues/3).
To circumvent this problem, multi-processing can be deactivated for the data setup by appending `+data.num_preprocessing_workers=0` to the training call.
This will lead to slightly longer preprocessing times, e.g. 4 minutes instead of 1 minute on QM9 using a modern CPU. However, this is not a big harm as the data setup only needs to be done once prior to training.
Note that the regular, repeated data loading is not impacted by this setting and can still run in parallel (it is configured with `data.num_workers`).

## Changes in this implementation

Compared to previous implementations of G-SchNet, we improved the scalability and simplified the adaptation to custom data sets.
The changes we made mainly concern the preparation of data and the reconstruction of the 3d positional distribution.
-   When sampling trajectories of atom placement steps for training, instead of using the molecular graph to determine the available neighbors of a focus atom we now employ a fixed radial cutoff called _placement cutoff_. All atoms within the placement cutoff are considered to be neighbors of the focus. Optionally, we allow to use covalent radii in this process. In that case, we check whether the covalent radii of the focus atom and atoms within the placement cutoff are overlapping to determine neighbors. 
-   When reconstructing the 3d positional distribution, we now only use distance predictions of atoms within a fixed, radial _prediction cutoff_ around the focus atom instead of using all previously placed atoms. This means that the number of distance distributions predicted by the model in each step is bound and can be controlled with the prediction cutoff, which improves the scaling of G-SchNet when applying it to larger molecules.

Accordingly, in comparison to previous implementations where G-SchNet had only a single _model cutoff_ that determined which atoms are exchanging messages in the SchNet interaction blocks, this version has three cutoffs as hyperparameters, namely the _model cutoff_, the _prediction cutoff_, and the _placement cutoff_.

## Citation

If you use G-SchNet in your research, please cite the corresponding publications:

N.W.A. Gebauer, M. Gastegger, S.S.P. Hessmann, K.-R. Müller, and K.T. Schütt. _Inverse design of 3d molecular structures with conditional generative neural networks_. Nature Communications 13, 973 (2022). https://doi.org/10.1038/s41467-022-28526-y

N. Gebauer, M. Gastegger, and K. Schütt. _Symmetry-adapted generation of 3d point sets for the targeted discovery of molecules_. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems 32, 7566–7578. Curran Associates, Inc. (2019). http://papers.nips.cc/paper/8974-symmetry-adapted-generation-of-3d-point-sets-for-the-targeted-discovery-of-molecules.pdf

K.T. Schütt, S.S.P. Hessmann, N.W.A. Gebauer, J. Lederer, and M. Gastegger. _SchNetPack 2.0: A neural network toolbox for atomistic machine learning_. arXiv preprint arXiv:2212.05517 (2022). https://arxiv.org/abs/2212.05517

    @Article{gebauer2022inverse,
        author={Gebauer, Niklas W. A. and Gastegger, Michael and Hessmann, Stefaan S. P. and M{\"u}ller, Klaus-Robert and Sch{\"u}tt, Kristof T.},
        title={Inverse design of 3d molecular structures with conditional generative neural networks},
        journal={Nature Communications},
        year={2022},
        volume={13},
        number={1},
        pages={973},
        issn={2041-1723},
        doi={10.1038/s41467-022-28526-y},
        url={https://doi.org/10.1038/s41467-022-28526-y}
    }
    @incollection{gebauer2019symmetry,
        author = {Gebauer, Niklas and Gastegger, Michael and Sch\"{u}tt, Kristof},
        title = {Symmetry-adapted generation of 3d point sets for the targeted discovery of molecules},
        booktitle = {Advances in Neural Information Processing Systems 32},
        editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
        year = {2019},
        pages = {7566--7578},
        publisher = {Curran Associates, Inc.},
        url = {http://papers.nips.cc/paper/8974-symmetry-adapted-generation-of-3d-point-sets-for-the-targeted-discovery-of-molecules.pdf}
    }
    @Article{schutt2022schnetpack,
      title={SchNetPack 2.0: A neural network toolbox for atomistic machine learning},
      author={Sch{\"u}tt, Kristof T and Hessmann, Stefaan SP and Gebauer, Niklas WA and Lederer, Jonas and Gastegger, Michael},
      journal={arXiv preprint arXiv:2212.05517},
      year={2022}
    }

## How does cG-SchNet work?

cG-SchNet is an autoregressive neural network.
It builds 3d molecules by placing one atom after another in 3d space.
To this end, the joint distribution of all atoms is factorized into single steps, where the position and type of the new atom depends on the preceding atoms (Figure a).
The model also processes conditions, i.e. values of target properties, which enable it to learn a conditional distribution of molecular structures.
This distribution allows targeted sampling of molecules that are highly likely to exhibit specified conditions (see e.g. the distribution of the polarizability of molecules generated with cG-SchNet using five different target values in Figure b).
The type and absolute position of new atoms are sampled successively, where the probability of the positions is apporximated from predicted pairwise distances to preceding atoms.
In order to improve the accuracy of the approximation and steer the generation process, the network uses two auxiliary tokens, the focus and the origin.
The new atom always has to be a neighbor of the focus and the origin marks the supposed center of mass of the final structure.
A scheme explaining the generation procedure can be seen in Figure c.
It uses 2d positional distributions for visualization purposes.
For more details, please refer to the [cG-SchNet publication](https://www.nature.com/articles/s41467-022-28526-y).  

![generated molecules](https://github.com/atomistic-machine-learning/cG-SchNet/blob/main/images/concept_results_scheme.png)
