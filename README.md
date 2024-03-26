# Estimating Functional and Interventional Connectivity in Neural Data 

![Estimating Functional and Interventional Connectivity in Neural Data](https://github.com/amin-nejat/FCF/assets/5959554/64054f9e-98e6-495c-bc47-72d5325f5e2c)

This code package implements a series of functional connectivity methods including information theory-based and attractor reconstruction-based measures, and introduces methods for computing interventional connectivity based on perturbed neural data. Codes for simulating time series from popular dynamical systems are implemented for exploratory purposes. 

See **[our paper](https://openreview.net/forum?id=3ucmcMzCXD)** for further details:


```
@article{PhysRevResearch.5.043211,
  title = {Predicting the effect of micro-stimulation on macaque prefrontal activity based on spontaneous circuit dynamics},
  author = {Nejatbakhsh, Amin and Fumarola, Francesco and Esteki, Saleh and Toyoizumi, Taro and Kiani, Roozbeh and Mazzucato, Luca},
  journal = {Phys. Rev. Res.},
  volume = {5},
  issue = {4},
  pages = {043211},
  numpages = {14},
  year = {2023},
  month = {Dec},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevResearch.5.043211},
  url = {https://link.aps.org/doi/10.1103/PhysRevResearch.5.043211}
}
```
**Note:** This research code remains a work-in-progress to some extent. It could use more documentation and examples. Please use at your own risk and reach out to us (anejatbakhsh@flatironinstitute.org) if you have questions. If you are using this code package, please cite our paper.

## A short and preliminary guide

### Installation Instructions

1. Download and install [**anaconda**](https://docs.anaconda.com/anaconda/install/index.html)
2. Create a **virtual environment** using anaconda and activate it

```
conda create -n fcf python=3.8
conda activate fcf
```

3. Install fcf package

```
git clone https://github.com/amin-nejat/fcf.git
cd fcf
pip install -r requirements.txt 
pip install -e .
```

4. Run demo file

```
python demo.py
```

Since the code is preliminary, you will be able to use `git pull` to get updates as we release them.
