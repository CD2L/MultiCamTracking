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
from torch.utils.tensorboard import SummaryWriter

nb_logs = len(os.listdir('runs/siamese-network'))
writer = SummaryWriter(f'runs/siamese-network/exp{nb_logs}')

def distance(x, y):
    return torch.sum((x - y) ** 2, dim=1)

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Calculates the triplet loss given three input tensors, anchor, positive, and negative.
    :param anchor: The anchor tensor with shape (batch_size, embedding_dim).
    :param positive: The positive tensor with shape (batch_size, embedding_dim).
    :param negative: The negative tensor with shape (batch_size, embedding_dim).
    :param margin: The margin value to use in the triplet loss calculation.
    :return: The triplet loss value.
    """
    positive_distance = distance(anchor, positive)
    negative_distance = distance(anchor, negative)
    loss = torch.clamp(positive_distance - negative_distance + margin, min=0.0)
    return torch.mean(loss)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

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
    with open("args.yml", "r") as f:
        args = yaml.safe_load(f)

    model.train()
    train_loss = 0.0
    
    for batch_idx, (anc, pos, neg) in enumerate(tqdm(dataloader)):
        anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)

        out_anc, out_pos, out_neg = model(anc), model(pos), model(neg)
        optimizer.zero_grad()
        loss = loss_fn(out_anc, out_pos, out_neg).unsqueeze(0)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_loss += loss.item()
            
    train_loss /= len(dataloader)
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('lr', get_lr(optimizer), epoch)

    print(
        f"train epoch {epoch}/{args['num_epochs']}",
        f"loss epoch {train_loss:.5f}"
    )

    return train_loss

def test(model, loss_fn, dataloader, device, epoch):
    model.eval()
    test_loss = 0.0

    for batch_idx, (anc, pos, neg) in enumerate(dataloader, 0):
        anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)

        with torch.no_grad():
            out = model(anc), model(pos), model(neg)

            loss = loss_fn(*out)
            test_loss += loss.item()

    test_loss /= len(dataloader)
    writer.add_scalar('Loss/test', test_loss, epoch)

    print(
        f"eval ",
        f"loss {test_loss:.5f}"
    )

    return test_loss

def save_example(epoch, x, y, z, similarity_y, similarity_z):
    x = x.permute(0,2,3,1)
    y = y.permute(0,2,3,1)
    z = z.permute(0,2,3,1)

    fig, axes = plt.subplots(nrows=len(x), ncols=3)
    fig.tight_layout
    
    axes[0][0].set_title('anchor', pad=20)
    axes[0][1].set_title('pos', pad=20)
    axes[0][2].set_title('neg', pad=20)

    for row,ax in enumerate(axes):
        ax[0].imshow(x[row])
        ax[0].axis("off")
        
        ax[1].imshow(y[row])
        ax[1].text(12,0,'%.2f'%torch.sum(similarity_y[row]).item())
        ax[1].axis("off")
        
        ax[2].imshow(z[row])
        ax[2].text(12,0,'%.2f'%torch.sum(similarity_z[row]).item())
        ax[2].axis("off")   

    if not os.path.exists('results'):
        os.mkdir('results')

    plt.savefig(f"results/sample_epoch_{epoch}.jpg", dpi=800)
    plt.close()

if __name__ == '__main__':
    img = cv.imread('Market-1501-v15.09.15/query/0004_c2s3_059152_00.jpg', cv.IMREAD_ANYCOLOR)
    arr = img.reshape((1, *img.shape))
    
    arr = np.append(arr, [img, img], axis=0)

    arr = arr.transpose(0,3,1,2)

    arr = torch.from_numpy(arr)
    save_example('test', arr, arr, arr)