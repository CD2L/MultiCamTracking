import math
import numpy as np
from tqdm import tqdm
import yaml
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

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

            loss = loss_fn(**out)
            test_loss = loss.item()

    test_loss /= len(dataloader)

    print(
        f"eval ",
        f"loss {test_loss:.5f}"
    )

    return test_loss