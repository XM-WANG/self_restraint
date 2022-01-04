## Model & Training

Model and training related cheat sheet.

#### 1. Torch model
```python
import torch
import torch.nn as nn

class WzzModel(nn.Module):
    def __init__(self,config):
        super(WzzModel, self).__init__()
        self.config = config
        self.input_layer = nn.Linear(self.config['input_size'],self.config['hidden_size'], bias=False)
        self.output_layer = nn.Linear(self.config['hidden_size'],self.config['output_size'])
    def forward(self, feat):
        output = self.input_layer(feat)
        output = self.output_layer(output)
        return output
```

#### 2. clip_grad_norm_()
```python
import torch.nn as nn

loss.backward()
nn.utils.clip_grad_norm_(model.parameters(), 1)
optimizer.step()
```

#### 3. torch.save
```python
import torch

model_path = "./model.pkl"
torch.save(model.state_dict(), model_path)
```

#### 4. load_state_dic() & torch.load()
```python 
import torch

model_path = "./model.pkl"
params = torch.load(model_path, map_location='cpu')
model.load_state_dict(params, strict=True)
```

#### 5. args
```python 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5)
parser.add_argument('--with-pretrained', dest="pretrained", action='store_true')
parser.add_argument('--no-pretrained', dest="pretrained", action='store_false')
parser.set_defaults(pretrained=False)
args = parser.parse_args()
```