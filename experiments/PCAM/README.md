### Folders

* `data`			: Contains the PatchCamelyon dataset, see the link inside this folder for where to download the data
* `experiments` 	: Contains the experiment specifications. These files are checked by the script "run_experiments.py".
	- `1_todo.txt` 		 : contains the experiments to be run (see e.g. 1_todo_se2.txt for the set of experiments used in the paper (where the average of 3 runs are reported))
	- `2_inprogress.txt` : contains the experiment that is currently being run. In case the server is shutdown during training, the script picks up from where it was interrupted.
	- `3_finished.txt`	 : this is to archive to complete experiments
* `models`		: Contains the model specifications. These files are read in the main script "train.py" and are based on the g-splinets package.
* `results`		: The trained models are exported to this directory. Results are archived per model (each model gets its own sub-directory) and each trained model again gets a sub-directory with a time stamp.

### Files
* `pcam.py` 			: the code used for reading in the PCam dataset
* `load_data.ipynb`		: a jupyter notebook file that demos how to read the PCam data
* `run_experiments.py` 	: script that runs a set of experiments (see also folder structure). Called via "python3 ./run_experiments.py" or "python ./run_experiments.py"
* `run_experiments.sh` 	: a linux script file that demos how to run train.py
* `train.py` 			: trains a model. See that last section of this file for options. And see e.g. run_experiments.py for usage.
