# Equivariant Transformer

## Cloning this repo
To clone this project, run `git clone --recurse-submodules https://github.com/akosiorek/eqv_transformer`.
If you cloned without the `--recurse-submodules` option, then run `git submodule update --init --recursive`.

## Dependencies

If you execute `setup_virtualenv.sh`, it will create a virtual environment and install all required dependencies. Alternatively, you can install all the dependencies using `pip install -r requirements.txt`.

## Training a model

Example command to train a model (in this case the Set Transformer on the constellation dataset):
```
python3 scripts/train.py --data_config configs/constellation.py --model_config configs/set_transformer.py --run_name my_experiment --learning_rate=1e-4 --batch_size 128
```

The model and the dataset can be chosen by specifying different config files. Flags for configuring the model and
the dataset are available in the respective config files. The project is using
[forge](https://github.com/akosiorek/forge) for configs and experiment management. Please refer to 
[this forge description](http://akosiorek.github.io/ml/2018/11/28/forge.html) and 
[examples](https://github.com/akosiorek/forge/tree/master/forge/examples) for details.

### Counting patterns in the constellation dataset

The first task implemented is counting patterns in the constellation dataset. We generate
a fixed dataset of constellations, where each constellation
consists of 0-8 patterns; each pattern consists of corners of a shape. Currently available shapes are triangle,
square, pentagon and an L. The task is to count the number of occurences of each pattern.
To save to file the constellation datasets, run before training:
```
python3 scripts/data_to_file.py
```
Else, the constellation datasets are regenerated at the beginning of the training.

#### Dataset and model consistency
When changing the dataset parameters (e.g. number of patterns, types of patterns etc) make sure that the model
parameters are adjusted accordingly. For example `patterns=square,square,triangle,triangle,pentagon,pentagon,L,L`
means that there can be four different patterns, each repeated two times. That means that counting will involve four
three-way classification tasks, and so that `n_outputs` and `output_dim` in `classifier.py` needs to be set to `4` and
`3`, respectively. All this can be set through command-line arguments. 


### QM9
```
python scripts/train_molecule.py \
    --run_name "molecule_homo" \
    --model_config "configs/molecule/eqv_transformer_model.py" \
    --model_seed 0
    --data_seed 0 \
    --task homo
```

### Hamiltonian dynamics

## Contributing

Contributions are best developed in separate branches. Once a change is ready, please submit a pull request with a
description of the change. New model and data configs should go into the `config` folder, and the rest of the code
should go into the `eqv_transformer` folder.