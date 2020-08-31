This folder contains executables for the modelbase backend.

### Files
 * `learn_initial_models.py`: an executable python script that trains a set of simple models
   and stores them in `fittel_models/`. 
 * `webservice.py`: an executable python script that starts the modelbase webservice.
 * `run_conf_defaults.cfg`: the default configuration file for `webservice.py`. Do not change this.
 * `run_conf.cfg`: the user configuration file for `webservice.py`. Use this instead of `run_conf_defaults.cfg`.
 * [deprecated] `profiling.py`: a helper for performance profiling.
 * `modelbase.wsgi`: a WSGI specification for the modelbase webservice .
  
### Folders
 * [to be moved] `experiments/`: to be moved.
 * [to be moved] `julien/`: to be moved.
 * `fitted_models/`: contains models that are by default loaded by `webservice.py`.  
 