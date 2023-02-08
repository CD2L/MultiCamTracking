import torch
import yaml
import os
from torch import nn
from src.model import SiameseModel
from src.dataset import ImageDataset, MultiAnglePersonDataset
from src.utils import train_test_split, train, test, triplet_loss, save_example, distance, get_lr

def main():
    torch.cuda.empty_cache()

    with open("args.yml", "r") as f:
        args = yaml.safe_load(f)

    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')

    show_sample = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MultiAnglePersonDataset(args['dataset_path'])

    train_dataset, test_dataset = train_test_split(dataset,0.8,args['batch_size'])

    model = SiameseModel()
    model = nn.DataParallel(model)
    model = model.to(device) 

    if args["pretrained"]:
        model.load_state_dict(torch.load(args['pretrained'])['model'])

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,200,200], gamma=0.1)

    loss_history = {"fit": [], "val": []}

    for epoch in range(1, args['num_epochs'] + 1):
        train_loss = train(epoch, model, triplet_loss, train_dataset, optimizer, device)
        test_loss = test(model, triplet_loss, test_dataset, device, epoch)

        scheduler.step()

        loss_history["fit"].append(train_loss)
        loss_history["val"].append(test_loss)

        if show_sample and not epoch % args['sample_checkpoints']:
            sample_loader = torch.utils.data.DataLoader(
                dataset, batch_size=3, shuffle=True, num_workers=0
            ) 

            data_iter = iter(sample_loader)
            x, y, z = next(data_iter)

            save_example(epoch, x, y ,z, distance(x,y), distance(x,z))

        if not epoch % args['model_checkpoints']:
             torch.save(
                {
                    "lr": get_lr(optimizer),
                    "loss_history": loss_history,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                f"checkpoints/checkpoint_{epoch}.pkl",
            )

if __name__ == '__main__':
    main()