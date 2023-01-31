import math
import numpy as np
from tqdm import tqdm
import yaml
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import cv2 as cv

with open("args.yml", "r") as f:
    args = yaml.safe_load(f)

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Calculates the triplet loss given three input tensors, anchor, positive, and negative.
    :param anchor: The anchor tensor with shape (batch_size, embedding_dim).
    :param positive: The positive tensor with shape (batch_size, embedding_dim).
    :param negative: The negative tensor with shape (batch_size, embedding_dim).
    :param margin: The margin value to use in the triplet loss calculation.
    :return: The triplet loss value.
    """
    positive_distance = torch.sum((anchor - positive) ** 2, dim=1)
    negative_distance = torch.sum((anchor - negative) ** 2, dim=1)
    loss = torch.clamp(positive_distance - negative_distance + margin, min=0.0)
    return torch.mean(loss)

def train_test_split(dataset, train_size, batch_size, shuffle=True):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(math.floor(train_size * dataset_size))

    if shuffle:
        np.random.shuffle(indices)
    
    train_sampler, test_sampler = (
        SubsetRandomSampler(indices[:split]),
        SubsetRandomSampler(indices[split:])
    )

    return (
        DataLoader(
            dataset, batch_size=batch_size, num_workers=0, sampler=train_sampler
        ),
        DataLoader(
            dataset, batch_size=batch_size, num_workers=0, sampler=test_sampler
        )
    )

def train(epoch, model, loss_fn, dataloader, optimizer, device):
    model.train()
    train_loss = 0.0
    
    for batch_idx, (anc, pos, neg) in enumerate(tqdm(dataloader)):
        anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)

        out_anc, out_pos, out_neg = model(anc, pos, neg)
        optimizer.zero_grad()
        loss = loss_fn(out_anc, out_pos, out_neg).unsqueeze(0)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_loss += loss.item()

    train_loss /= len(dataloader)

    print(
        f"train epoch {epoch}/{args['num_epochs']}",
        f"loss epoch {train_loss:.5f}"
    )

    return train_loss

def test(model, loss_fn, dataloader, device):
    model.eval()
    test_loss = 0.0

    for batch_idx, (anc, pos, neg) in enumerate(dataloader, 0):
        anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)

        with torch.no_grad():
            out = model(anc,pos,neg)

            loss = loss_fn(*out)
            test_loss = loss.item()

    test_loss /= len(dataloader)

    print(
        f"eval ",
        f"loss {test_loss:.5f}"
    )

    return test_loss

def save_example(epoch, x,y,z):
    x = x.permute(0,2,3,1)
    y = y.permute(0,2,3,1)
    z = z.permute(0,2,3,1)

    fig, axes = plt.subplots(nrows=len(x), ncols=3)
    fig.tight_layout
    
    axes[0][0].set_title('anchor')
    axes[0][1].set_title('pos')
    axes[0][2].set_title('neg')

    for row,ax in enumerate(axes):
        ax[0].imshow(x[row], cmap="gray")
        ax[0].axis("off")
        
        ax[1].imshow(y[row], cmap="gray")
        ax[1].axis("off")
        
        ax[2].imshow(z[row], cmap="gray")
        ax[2].axis("off")

    if not os.path.exists('results'):
        os.mkdir('results')

    plt.savefig(f"results/sample_epoch_{epoch}.jpg", dpi=1200)
    plt.close()