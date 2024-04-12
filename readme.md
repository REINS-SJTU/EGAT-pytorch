# CodeAppendix

Code for paper "EGAT: Edge-Featured Graph Attention Network"

## Dependencies

in `requirements.txt`

## File Structure
- data: Directory for dataset.
- model:
  - node.py: The node module of EGAT.
  - edge.py: The edge module of EGAT.
  - mgcn.py: The edge and node modules of MGCN, including the EGAT_MGCN (`AttentionVertexModule`)
  - nnconv.py: The node module of NNConv, including the EGAT_NNConv (`AttentionNNConv`)
  - net.py: The network structure of EGAT, for both AMLSim (`AMLSimNet`) and citation networks (Cora, Citeseer and PubMed) (`CitationNet`). The structure of `CitationNet` is hard coded.
- trainer: The training process (see: pytorch-lightning) of AMLSim and citation networks.
- transforms: The transformers of dataset.
- dataset.py: Some of the preprocessing of AMLSim and all the preprocessing of citation networks.
- main.py: The entry file.
- config.yml: Hyperparameter config file.

## Usage

### Dataset Prepare
Please copy all dataset to `data` directory. (available at [this url](https://drive.google.com/drive/folders/1Q3VLfk3i81R-HEUfNHgMGA3h22u_F3Xq?usp=drive_link))

### Hyperparameters
You can control the hyperparameter in `config.yml`. where the meaning of each hyperparameter is commented .

### Train
Run `python main.py` to train the model. The results are reported in the terminal.
