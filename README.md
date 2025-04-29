## Quick Start
Clone the repo:
```
git clone git@github.com:LukeV37/PUMiNet_Training.git
```

Ensure python3 is installed, and setup the virtual environment:
```
source setup.sh
```

Please be patient while pip installs requirements...

## Training PUMiNet
Check the `datasets` directory and update `config.yaml` to point to the training dataset using the `in_sample` var. Then train the model with the following command:
```
python train.py
```
The models weights are saved after each epoch. After training, the loss curve is plotted. Please check the `out_path` var in `config.yaml`.

## Evaluate PUMiNet
Evaluate the model with the following command:
```
python eval.py
```
The results will be stored as histograms in the `out_path` var in `config.yaml`.

## Dependencies
<ul>
  <li>python3</li>
  <li>python3 -m venv</li>
</ul>
