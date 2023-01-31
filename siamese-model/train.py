import torch
import yaml
from torch import nn
from src.model import SiameseModel
from src.dataset import ImageDataset
from src.utils import train_test_split, train, test, triplet_loss

def main():
    torch.cuda.empty_cache()

    with open("args.yml", "r") as f:
        args = yaml.safe_load(f)

    show_sample = True
    dtype = torch.float
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ImageDataset(args['dataset_path'])

    train_dataset, test_dataset = train_test_split(dataset,0.8,args['batch_size'])

    model = SiameseModel()
    model = nn.DataParallel(model)
    model = model.to(device) 

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=0.1)

    loss_history = {"fit": [], "val": []}

    for epoch in range(1, args['num_epochs'] + 1):
        train_loss = train(epoch, model, triplet_loss, train_dataset, optimizer, device)
        test_loss = test(model, triplet_loss, test_dataset, device)

        loss_history["fit"].append(train_loss)
        loss_history["val"].append(test_loss)

if __name__ == '__main__':
    main()