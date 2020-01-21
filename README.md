## Spline G-CNNs

This repository contains the source code accompanying the paper "B-Spline CNNs on Lie Groups" which is published ICLR 2020 (https://openreview.net/forum?id=H1gBhkBFDH). The experiments performed in this paper are based on the `gsplinets_tf` library found in this repository. The full set of experiments can be reproduced using the scripts of the accompanying repository which can be found in the directory `experiments`.

## Folder structure
The folder structure is as follows:

* `gsplinets_tf` contains the main tensorflow library.

* `demo` includes some short jupyter notebook demo's on how to use the the code.

* `docs` contains basic documentation of the layer interfaces.

* `experiments` contains the scripts to run the experiments of the ICLR 2020 publication.

## Dependencies

This code as based on tensorflow and has been tested with the following library versions:

* tensorflow-gpu==1.15

* numpy==1.17.4

* scipy==1.3.2

* matplotlib==3.1.1

* jupyter==1.0.0

An appropriate environment may be constructed with conda via

```
conda create --yes --name tf
conda activate tf
conda install tensorflow-gpu==1.15 numpy==1.17.4 scipy==1.3.2 matplotlib==3.1.1 jupyter==1.0.0 --yes
```

## Cite

The development of this library was part of the work done for the paper "B-Spline CNNs on Lie groups" (https://openreview.net/forum?id=H1gBhkBFDH). Please cite this work if you use this code:

```
@inproceedings{
bekkers2020bspline,
title={B-Spline {\{}CNN{\}}s on Lie groups},
author={Erik J Bekkers},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=H1gBhkBFDH}
}
```

## License

The code and scripts in this repository are distributed under MIT license. See LICENSE file.
