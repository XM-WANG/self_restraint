## Tensor / Array Operation

Tensor related cheat sheet.

### numpy.concatenate()
```python
import numpy as np

feat = np.concatenate([feat_1,feat_2], axis=2) # dim of feat_1 and feat_2 should be same.
```

### numpy.transpose()
```python

feat = np.transpose(feat, (1, 0, 2)) # feat should be a 3-dim array
```

### torch.cat()
```python
import torch

feat = torch.cat([feat_1,feat_2], axis=2) # dim of feat_1 and feat_2 should be same.
```

### torch.transpose()
```python
import torch

# tensor.transpose()
feat_T1 = feat.transpose(0, 2) # feat should at least has 3-dim.

# torch.transpose()
feat_T2 = torch.transpose(feat,0,2)
```