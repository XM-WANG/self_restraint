## Conda Environment

#### Register kernel to jupyter
```
conda activate myenv
conda install nb_conda_kernels
python -m ipykernel install --user --name myenv --display-name "myenv"
conda install -n myenv ipykernel --update-deps --force-reinstall
```

#### Check jupyter kernel
```
jupyter kernelspec list
```

#### Vscode debug demo
```
launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {"PYTHONPATH": "${workspaceFolder}${pathSeparator}${env:PYTHONPATH}"},
            "args": [
                "--bsz","24",
                "--eval-size", "48",
                "--iters", "180",
                "--model-name", "bert-base-cased",
            ]
        }
    ]
}
```

#### Update Conda
```
conda update -n base -c defaults conda --repodata-fn=repodata.json
```
