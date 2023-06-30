import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import os

from VisualRL.common.utils import get_device, create_path_for_results
from VisualRL.vae.model import VAE
from VisualRL.vae.dataset import VaeImageDataset


DATA_SET_PATH = os.path.join(os.environ["VISUAL_PUSHING_HOME"], "images/all_objects_masks_random")
RESULTS_SAVE_PATH = os.path.join(os.environ["VISUAL_PUSHING_HOME"], "results/vae")

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type = int, default = 64)
parser.add_argument('--epochs', type = int, default = 120)
parser.add_argument('--seed', type = int, default = 1)
parser.add_argument('--eval_freq', type = int, default = 5)
parser.add_argument('--device', type = str, default = 'auto')
args = parser.parse_args()

def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD


def main():
    # torch.manual_seed(args.seed)

    device = get_device(args.device)

    kwargs = {'num_workers': 1, 'pin_memory': True}
    # load data
    transform  = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
    train_dataset = VaeImageDataset(base_path = DATA_SET_PATH, split = True, transform = transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, pin_memory = False, num_workers = 2)
    test_dataset = VaeImageDataset(base_path = DATA_SET_PATH, train = False, split = True, transform = transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, pin_memory = False, num_workers = 2)
    print('\nall data loader! \n')
    # load model
    model = VAE(device = device, image_channels = 1, h_dim = 1024, z_dim = 6)
    optimizer = optim.Adam(model.parameters(), lr = 0.0003)
    criterion = nn.MSELoss()


    # train
    images_path, model_path = create_path_for_results(RESULTS_SAVE_PATH, image = True, model = True)
    writer = SummaryWriter(model_path)
    for epoch in range(args.epochs):
        train_loss = 0
        for batch_id, image in enumerate(train_loader):
            image = image.to(device)
            image_recon, z, mu, logvar = model(image)
            loss, bce, kld = loss_fn(image_recon, image, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch + 1, args.epochs, loss.item()/args.batch_size, bce.item()/args.batch_size, kld.item()/args.batch_size))
            writer.add_scalar('train/Total Loss', loss.item()/args.batch_size, epoch)
            writer.add_scalar('train/BCE', bce.item()/args.batch_size, epoch)
            writer.add_scalar('train/KLD', kld.item()/args.batch_size, epoch)
        # test
        with torch.no_grad():
            for batch_id, test_image, in enumerate(test_loader):
                test_image = test_image.to(device)
                test_recon, z, mu, logvar = model(test_image)
                loss, bce, kld = loss_fn(test_recon, test_image, mu, logvar)

            print("TEST Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch + 1, args.epochs, loss.item()/args.batch_size, bce.item()/args.batch_size, kld.item()/args.batch_size))
            writer.add_scalar('eval/Total Loss', loss.item()/args.batch_size, epoch)
            writer.add_scalar('eval/BCE', bce.item()/args.batch_size, epoch)
            writer.add_scalar('eval/KLD', kld.item()/args.batch_size, epoch)
        # save the testing results
        image_sample = image[0]
        image_recon_sample = image_recon[0]
        test_sample = test_image[0]
        test_recon_sample = test_recon[0]
        saved_image = torch.stack([image_sample, image_recon_sample,
                                    test_sample, test_recon_sample])
        file_name = "epoch_{}.png".format(epoch + 1)
        torchvision.utils.save_image(saved_image, os.path.join(images_path, file_name), nrow = 2)

        # save model
        model.save(model_path, epoch)


if __name__ == '__main__':
    main()











