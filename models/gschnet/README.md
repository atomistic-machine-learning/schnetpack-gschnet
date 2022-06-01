# Conditional G-SchNet - A generative neural network for 3d molecules

Here we provide a re-implementation of [cG-SchNet](https://www.nature.com/articles/s41467-022-28526-y) using the most up-to-date version of [SchNetPack](https://github.com/atomistic-machine-learning/schnetpack/tree/dev).
Compared to previous versions, SchNetPack changed from batching molecules to batching atoms, effectively removing the need for padding individual systems.
G-SchNet greatly benefits from this change in terms of memory requirements, allowing to train models of the same expressivity on GPUs with less VRAM.

Furthermore, we changed a few details in this implementation concerning the model as well as the data pre-processing. For reproduction of the reported results, please refer to the specific repositories where we provide the code used in each publication:
-  [G-SchNet](https://github.com/atomistic-machine-learning/G-SchNet) ([Symmetry-adapted generation of 3d point sets for the targeted discovery of molecules, 2019](http://papers.nips.cc/paper/8974-symmetry-adapted-generation-of-3d-point-sets-for-the-targeted-discovery-of-molecules))
- [cG-SchNet](https://github.com/atomistic-machine-learning/cG-SchNet) ([Inverse design of 3d molecular structures with conditional generative neural networks, 2022](https://www.nature.com/articles/s41467-022-28526-y))

_**Disclaimer**: The switch to the new SchNetPack version required us to rewrite almost the entire code base. Although the training is working and trained models successfully sample from learned, conditional distributions, we are still thoroughly investigating the code and running experiments to verify the implementation. If you encounter bugs or unexpected behaviour, please file an issue. Accordingly, we expect that we might have some breaking updates/changes in the near future. We aim for a first stable release with the release of SchNetPack 2.0._

### Citation

If you use G-SchNet in your research, please cite the corresponding publications:

N.W.A. Gebauer, M. Gastegger, S.S.P. Hessmann, K.-R. Müller, and K.T. Schütt. _Inverse design of 3d molecular structures with conditional generative neural networks_. Nature Communications 13, 973 (2022). https://doi.org/10.1038/s41467-022-28526-y

N. Gebauer, M. Gastegger, and K. Schütt. _Symmetry-adapted generation of 3d point sets for the targeted discovery of molecules_. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems 32, 7566–7578. Curran Associates, Inc. (2019).

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

### Installation

**ToDo**

# Getting started

**ToDo**
