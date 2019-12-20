## Spline G-CNNs

This repository contains the sore code accompanying the paper "B-Spline CNNs on Lie Groups" which is published ICLR 2020 (https://openreview.net/forum?id=H1gBhkBFDH).

## Folder structure
The folder structure is as follows:

* The main library is found in the folder `gsplinets`. 

* See `demo` for short jupyter notebook demo's on how to use the the code.

* Also see `docs` for the documentation of the layer interfaces.

The paper also describes a series of experiments. The scripts that produce the results of the paper, and which are based on the library present in this repo, will appear soon in a separate repository.

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

*gsplinets* is distributed under MIT license. See LICENSE file.
