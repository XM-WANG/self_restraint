## Tensor / Array Operation

Tensor related cheat sheet.

### numpy.concatenate()

```python
import numpy as np

feat_1 = np.random.rand(10,20,3)
feat_2 = np.random.rand(10,20,5)
feat = np.concatenate([feat_1,feat_2], axis=2)
```

### numpy.transpose()

```python
import numpy as np
feat1 = np.random.rand(1, 2, 3)
feat2 = np.transpose(feat1, (1, 0, 2))
```

### torch.cat()
```python
import torch

feat_1 = torch.rand(10,20,3)
feat_2 = torch.rand(10,20,5)
feat = torch.cat([feat_1,feat_2], axis=2)
```

### torch.transpose()
```python
import torch

feat = torch.rand(0, 1, 2, 3)

# tensor.transpose()
feat_T1 = feat.transpose(0, 2)

# torch.transpose()
feat_T2 = torch.transpose(feat,0,2)
```