<img align="right" width="40%" height="40%" src="https://github.com/je-santos/PoreFlow-Net/blob/master/ilustrations/flow.PNG"/>

# BernFlow-Net
Implementation of
**BernFlow-Net: a 3D convolutional neural network**
## Instructions
1. Download the desired data from [my google drive](https://drive.google.com/drive/folders/1iQzeNL1BSOeh5repwVlkM0y4sZyeh6Kd?usp=sharing) (or create your own via your preferred simulation method)
2. Create a new conda env
```bash
conda create --name bernie_env python=3.6 keras-gpu matplotlib spyder
conda activate bernie_env
pip install git+https://github.com/je-santos/livelossplot
pip install hdf5storage
```

3. Use the train.py script to train a model. We can play with the features that go in, to assess which are the most relevant.

## Model architecture
This is how our network looks like:
![architecture](https://github.com/je-santos/PoreFlow-Net/blob/master/ilustrations/net.JPG)
## Methodology
![Process Overview](https://github.com/je-santos/PoreFlow-Net/blob/master/ilustrations/workflow.PNG)


The rest of the necessary packages should be available via pip

## Data
The full publication and all the training/testing data can be found [here](https://drive.google.com/drive/folders/1iQzeNL1BSOeh5repwVlkM0y4sZyeh6Kd?usp=sharing). An excel file is provided with the list of samples available.  


## Room for improvement
The keras tunner could be used to optimize the number of filters on each encoding branch

## Collaborations
We welcome collaborations


## Citation
If you use our code for your own research, we would be grateful if you cite our publication
[AWR](https://www.sciencedirect.com/science/article/pii/S0309170819311145)
```
@article{PFN2020,
title = "PoreFlow-Net: a 3D convolutional neural network to predict fluid flow through porous media",
journal = "Advances in Water Resources",
pages = "103539",
year = "2020",
issn = "0309-1708",
doi = "https://doi.org/10.1016/j.advwatres.2020.103539",
url = "http://www.sciencedirect.com/science/article/pii/S0309170819311145",
author = "Javier E. Santos and Duo Xu and Honggeun Jo and Christopher J. Landry and Maša Prodanović and Michael J. Pyrcz",
}
```
