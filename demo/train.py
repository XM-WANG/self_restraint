import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import load_data,setup_seed,setup_logs
from models import Demo

def train(model,trainloader,device,optimizer,criterion):
    """ Train model in a single epoch. This is a classification demo.
    """
    running_loss,correct,total = 0.0,0,0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs,labels = inputs.to(device),labels.to(device)
        outputs = model(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.cpu().item()
    return correct/total, running_loss

def test(model,testloader,device):
    """ Test model in a single epoch. This is a classification demo.
    """
    correct,total = 0,0
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        inputs,labels = inputs.to(device),labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return correct/total

parser = argparse.ArgumentParser()
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
parser.add_argument('-ep', '--max_epoch', type=int, default=10)
parser.add_argument('-wd','--weight_decay', type=float, default=5e-5)
parser.add_argument('-pt','--pretrained', type=bool, default=True)
parser.add_argument('-bs','--batch_size', type=int, default=256)
parser.add_argument('-opt','--optimizer', type=str, default='adam')
parser.add_argument('-sd','--seed', type=int, default=10)
args = parser.parse_args()

# Param
learning_rate = args.learning_rate
weight_decay = args.weight_decay
batch_size = args.batch_size
max_epoch = args.max_epoch
seed = args.seed
optimizer = args.optimizer
pretrained = args.pretrained
arg_dict = {"lr":learning_rate,"wd":weight_decay,"bs":batch_size,"me":max_epoch,"sd":seed,'opt':optimizer,"pt":pretrained}
setup_seed(seed)
logger, log_path, exp_path = setup_logs("logs/encoder",arg_dict)
logger.info(",".join([f"{str(k)}:{str(v)}" for k,v in arg_dict.items()]))
device_idx = 0
device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")

# Load & Transform Data
trainloader, testloader = load_transform_data(batch_size)

# Model
model = models.Demo()
model.to(device)

# Optimize
criterion = nn.CrossEntropyLoss()
if optimizer == 'adam':
    encoder_optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
elif optimizer == 'sgd':
    encoder_optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

for epoch in range(max_epoch):
    t1 = time.time()
    train_acc,loss = train(model,trainloader,device,encoder_optimizer,criterion)
    test_acc = test(model,testloader,device)
    t2 = time.time()
    logger.info(f"epoch[{epoch}] | time : {t2-t1:.2f} | loss : {loss:.2f} | Acc@train : {train_acc:.2f} | Acc@test : {test_acc:.2f}")
logger.info('Finished Training')
torch.save(model.state_dict(), exp_path)
