import torch
import matplotlib.pyplot as plt
from utils import Dataset, load_data
from trainer import batch_gd, evaluate
from models import DeepLOB

config = {
    # Data configs
    'data_path': './data/',
    'batch_size': 32,
    'num_classes': 3,
    'T': 100,
    'k': 10,

    # Training configs
    'lr': 0.01,
    'eps': 1.0,
    'epochs': 100,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'patience': 20,
    'min_delta': 1e-6,
    'print_freq': 10
}
print(config['device'])

for k in [10, 20, 50]:
    print(f'Training for the model with k={k}')
    config['k'] = k
    # Load the data
    train, val, test = load_data(config['data_path'])

    dataset_train = Dataset(data=train, k=config['k'], num_classes=config['num_classes'], T=config['T'])
    dataset_val = Dataset(data=val, k=config['k'], num_classes=config['num_classes'], T=config['T'])
    dataset_test = Dataset(data=test, k=config['k'], num_classes=config['num_classes'], T=config['T'])

    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=config['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=config['batch_size'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=config['batch_size'], shuffle=False)

    print(f'Training data shape: {dataset_train.x.shape} | Training label shape: {dataset_train.y.shape}')

    # Initialize the model, loss and optimizer
    model = DeepLOB(y_len=dataset_train.num_classes)
    model.to(config['device'])
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=config['eps'])

    # Train the model and print the classification report
    model, tr_loss, val_loss, tr_acc, vl_acc = batch_gd(model, criterion, optimizer, train_loader, val_loader, config)
    evaluate(model, test_loader, config)

    # Save the plot of training and validation loss and accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(tr_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss curves')
    plt.subplot(1, 2, 2)
    plt.plot(tr_acc, label='Training Accuracy')
    plt.plot(vl_acc, label='Validation Accuracy')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy curves')
    plt.tight_layout()
    plt.savefig(f'./Experiments/k_{k}_loss_acc.png')
