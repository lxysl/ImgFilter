import os
import datetime
import random

import torch
import numpy as np
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Sequential
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torchinfo import summary
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# from tqdm.notebook import tqdm
from tqdm import tqdm
import wandb

from dataset import pairDataset
from metric import ArcFace

device = torch.device('mps')


def setup_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def train_steps(loop, net, criterion, optimizer, metric, config):
    train_loss = []
    net.train()
    for step_index, (X1, X2, y) in loop:
        X1, X2, y = X1.to(device), X2.to(device), y.to(device)
        embedding1 = net(X1)
        embedding2 = net(X2)
        embedding = embedding1 + embedding2
        theta = metric(embedding, y)
        loss = criterion(theta, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        train_loss.append(loss)
        loop.set_postfix(loss=loss)
    return {"loss": np.mean(train_loss)}


# def val_steps(loop, net, criterion, config):
#     val_loss = []
#     val_acc = []
#     net.eval()
#     with torch.no_grad():
#         for step_index, (X, y) in loop:
#             X, y = X.to(device), y.to(device)
#             pred = net(X)
#             loss = criterion(pred, y).item()
#
#             val_loss.append(loss)
#             pred_result = torch.argmax(pred, dim=1).detach().cpu().numpy()
#             y = y.detach().cpu().numpy()
#             acc = accuracy_score(y, pred_result)
#             val_acc.append(acc)
#             loop.set_postfix(loss=loss, acc=acc)
#     return {"loss": np.mean(val_loss),
#             "acc": np.mean(val_acc)}


def train_epochs(train_dataloader, val_dataloader, net, criterion, optimizer, metric, config):
    num_epochs = config['num_epochs']
    train_loss_ls = []
    # train_loss_acc = []
    # val_loss_ls = []
    # val_loss_acc = []
    for epoch in range(num_epochs):
        train_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        # val_loop = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        train_loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')
        # val_loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')

        train_metrix = train_steps(train_loop, net, criterion, optimizer, metric, config)
        # val_metrix = val_steps(val_loop, net, criterion, config)

        train_loss_ls.append(train_metrix['loss'])
        # val_loss_ls.append(val_metrix['loss'])
        # val_loss_acc.append(val_metrix['acc'])

        print(f'Epoch {epoch + 1}: '
              f'train loss: {train_metrix["loss"]}; ')
        # print(f'Epoch {epoch + 1}: '
        #       f'val loss: {val_metrix["loss"]}; '
        #       f'val acc: {val_metrix["acc"]}')

        wandb.log({"train loss": train_metrix["loss"], })
        # "validation loss": val_metrix["loss"],
        # "validation accuracy": val_metrix["acc"]})
    return {'train_loss': train_loss_ls, }
    # 'val_loss': val_loss_ls,
    # 'val_acc': val_loss_acc}


def train(model_path, train_dataloader, val_dataloader, net, optimizer, criterion, metric, config):
    for train_X1, train_X2, train_y in train_dataloader:
        print("Shape of test X1: ", train_X1.shape)
        print("Shape of test X2: ", train_X2.shape)
        print("Shape of test y: ", train_y.shape, train_y.dtype)
        break
    # summary(net, (config['batch_size'], config['dim']), col_names=["input_size", "kernel_size", "output_size"],
    #         verbose=2)

    metrix = train_epochs(train_dataloader, val_dataloader, net, criterion, optimizer, metric, config)
    torch.save(net.state_dict(), model_path + '.pt')
    return metrix


if __name__ == '__main__':
    cur_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    cfg = {
        "seed": 42,
        "batch_size": 32,
        "class_num": 2,
        "embed": 2048,
        "model": "densenet_model",
        "num_epochs": 100,
        "optimizer": "SGD",
        "lr": 0.01,
        "lr_step": 50,
    }
    wandb.init(project="cloth_pair", name=cur_time, config=cfg)

    print("Using {} device".format(device))
    RANDOM_SEED = cfg["seed"]
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    log_dir = './logs/' + cur_time + '-' + cfg["model"]
    model_path = './model/' + cur_time + '-' + cfg["model"]

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(512, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    img_pairs = []
    labels = []
    with open('./dataset/positive_pair.txt') as f:
        for line in f.readlines():
            line = line.strip()
            img_pairs.append((line.split(',')[0], line.split(',')[1]))
            labels.append(1)
    with open('./dataset/negative_pair.txt') as f:
        for line in f.readlines():
            line = line.strip()
            img_pairs.append((line.split(',')[0], line.split(',')[1]))
            labels.append(0)
    test_ratio = 0.05
    train_img_pairs, test_img_pairs, train_labels, test_labels = train_test_split(img_pairs,
                                                                                  labels,
                                                                                  test_size=test_ratio,
                                                                                  random_state=RANDOM_SEED)
    train_dataset = pairDataset(train_img_pairs, train_labels, train_transform)
    test_dataset = pairDataset(test_img_pairs, test_labels, test_transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)

    net = resnet50(weights='DEFAULT').to(device)
    # make the fc layer as flatten layer
    net.fc = nn.Flatten()  # 2048*1*1 -> 2048

    optimizer = torch.optim.SGD(params=net.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['lr_step'], gamma=0.1)
    criterion = CrossEntropyLoss().to(device)
    metric = ArcFace(cfg["embed"], cfg["class_num"]).to(device)
    metrix = train(model_path, train_loader, test_loader, net, optimizer, criterion, metric, cfg)
    wandb.finish()
