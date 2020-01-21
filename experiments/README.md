## Spline G-CNNs

This part of the repository contains all the scripts and files used to produces the results described in the experimental part of the paper "B-Spline CNNs on Lie Groups" which is published ICLR 2020 (https://openreview.net/forum?id=H1gBhkBFDH).

## Folder structure
The folder structure is as follows:

* `CelebA` contains the scripts and data to run the landmark detection experiments.

* `PCAM` contains the scripts and data to run the Patch Camelyon experiments.

Both folders contain a file `run_experiments.py` which runs through a batch of experiments described in `experiments\todo.txt` within each experiment folder (see the `README.md` file in these folders). The `gsplinets_tf` library is inported via a relative path structure.

## Dependencies

See the README file in the top directory.

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

The code and scripts in this repository are distributed under MIT license. See LICENSE file in the top directory.
