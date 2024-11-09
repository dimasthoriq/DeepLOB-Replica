import torch

from utils import Dataset, load_data
from trainer import batch_gd, evaluate
from models import DeepLOB

# Load the data
data_path = './data/'
train, val, test = load_data(data_path)

batch_size = 32
k = 10
num_classes = 3
T = 100

dataset_train = Dataset(data=train, k=k, num_classes=num_classes, T=T)
dataset_val = Dataset(data=val, k=k, num_classes=num_classes, T=T)
dataset_test = Dataset(data=test, k=k, num_classes=num_classes, T=T)

train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

print(dataset_train.x.shape, dataset_train.y.shape)