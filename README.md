# Installation Instructions for the Functional Causal Flow [**Paper**](https://www.biorxiv.org/content/10.1101/2020.11.23.394916v2.abstract)

1. Download and install [**anaconda**](https://docs.anaconda.com/anaconda/install/index.html)
2. Create a **virtual environment** using anaconda and activate it

```
conda create -n ccm
conda activate ccm
```

3. Install CCM package

```
git clone https://github.com/amin-nejat/CCM.git
cd CCM
conda install --file requirements.txt 
pip install -e .
```