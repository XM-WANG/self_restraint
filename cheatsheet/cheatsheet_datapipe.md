## Data Pipe

Data related cheat sheet.

#### Numpy
```python
import numpy as np

with open("data.npy", "wb") as f:
    np.save(f, feat_1)
    np.save(f, feat_2)

with open("data.npy", "rb") as f:
    feat_1 = np.load(f)
    feat_2 = np.load(f)
```

#### Pickle
```python
import pickle as pkl

# save
with open(save_path,"wb") as f:
    pkl.dump(pkl_data,f)

# load
with open(save_path,"rb") as f:
    data = pkl.load(f)

train, test = data['train'], data['test'] # suppose data is a dict.
```

#### h5py
```python
import h5py

with h5py.File(f"data.hdf5","w") as f:
    f.create_dataset("feat_1",data=feat_1)
    f.create_dataset("feat_2",data=feat_2)

with h5py.File("data.hdf5","r") as f:
    feat_1 = f['feat_1'][()]
    feat_2 = f['feat_2'][()]

```

#### TensorDataset & DataLoader
```python
# Dependency
from torch.utils.data import DataLoader,TensorDataset
import torch
import numpy as np
# Key
input_ids, attention_mask, labels = np.random.rand(10,20,30),np.random.rand(10,20,30),np.random.rand(10)
trainingDataset = TensorDataset(torch.from_numpy(input_ids),torch.from_numpy(attention_mask),torch.from_numpy(labels))
trainingDataloader = DataLoader(trainingDataset,batch_size=3,shuffle=True)
# Demo
for input_ids, attention_mask, labels in trainingDataloader:
    ### some operations ###
    break
```

#### Pandas Display Setting
```python
pd.set_option('display.max_colwidth', 30)
pd.set_option('display.width', 200)
```

#### train_test_split
```python
import numpy as np
from sklearn.model_selection import train_test_split

feat1, feat2, feat3 = np.random.rand(10,20,30),np.random.rand(10,2),np.random.rand(10)
split_features = train_test_split(feat1, feat2, feat3, test_size=0.2, random_state = 42)
f1_train, f1_test,f2_train, f2_test,f3_train, f3_test = split_features
```
