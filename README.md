# ProDTI: Multi-scale Protein Representation with Cross-domain Adaption for Drug-Target Prediction

## Installation

Use Conda to set up the environment and install the required dependencies:

```bash
# Create a new conda environment
conda create --name prodti python=3.8
conda activate prodti

# Install PyTorch and dependencies
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
conda install -c dglteam dgl-cuda10.2==0.7.1
conda install -c conda-forge rdkit==2021.03.2
pip install dgllife==0.2.8
pip install -U scikit-learn
pip install yacs
pip install prettytable

# Clone the ProDTI repository
git clone https://github.com/htb130/TPI.git
cd ProDTI
```


## Running the Model

Before running, make sure to set up the correct `.yaml` configuration files for each task.

### In-domain Experiments (vanilla ProDTI)

`${dataset}` can be `bindingdb`, `biosnap`, or `human`  
`${split_task}` can be `random` or `cold`

```bash
python main.py --cfg "configs/ProDTI.yaml" --data ${dataset} --split ${split_task}
```
=======
# TPI
>>>>>>> 8813dc71e73153c2c2dde24e9d84ef4cda6f9baa
