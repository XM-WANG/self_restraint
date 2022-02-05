## Conda Environment

#### Register kernel to jupyter
```
conda activate myenv
conda install nb_conda_kernels
python -m ipykernel install --user --name myenv --display-name "myenv"
```

#### Check jupyter kernel
```
jupyter kernelspec list
```