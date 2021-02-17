# Self Restraint

This repo aim to quickly develop a new project with a fixed standard and elegant style. (All standards and styles are defined by WANG Xiaoming and will not be challenged by anyone else.)

## Directory Structure

Project should follow the dirctory structure below:

```
|-- root
    |-- .gitignore
    |-- README.md
    |-- LICENSE
    |-- models.py
    |-- train.py
    |-- utils.py
    |-- data
    |   |-- data_n
    |-- logs
    |   |-- time_stamp_n
    |   |   |-- log_n.log
    |   |   |-- model_n.pkl
    |-- scripts
    |   |-- run.py
    |-- temp
```
- Files: `trian.py`, `utils.py`, `models.py`.
- Folders:
  - `data`: The place to save data. It can be devided into sub-folders to save processed data.
  - `logs`: The place to save logs.
  - `scripts`: The place to save scrips. (e.g. hyper-parameters searching scripts.)
  - `temp`: The place to save intermediate result, or some testing code.

Folder `demo` in this repo is a demonstration of the structure mentioned above. 

## Quick Develop

Suppose that you want to develop such directory in the `../test` folder, set parameter `-p` as `../test`.

```
python setup.py -p=../test
```
