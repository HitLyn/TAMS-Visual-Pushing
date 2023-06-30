import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from VisualRL.vae.dataset import VaeImageDataset

# kwargs = {"num_workers": 1, "pin_memory": True}
# transformer = transforms.Compose([transforms.ToTensor()])
# train_dataset = VaeImageDataset(DATA_PATH, train = True, split = True)
# test_dataset = VaeImageDataset(DATA_PATH, train = False, split = True)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 128, shuffle = True, **kwargs)
# test_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 128, shuffle = Ture, **kwargs)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), input.size(1), 1, 1)

class VAE(nn.Module):
    def __init__(self, device, image_channels = 3, h_dim = 1024, z_dim = 64):
        super(VAE, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(
                nn.Conv2d(image_channels, 32, kernel_size = 4, stride = 2, bias = True),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size = 4, stride = 2, bias = True),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size = 4, stride = 2, bias = True),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size = 4, stride = 2, bias = True),
                nn.ReLU(),
                Flatten(),)
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
                UnFlatten(),
                nn.ConvTranspose2d(h_dim, 32, kernel_size = 5, stride = 2, bias = True),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 32, kernel_size = 5, stride = 2, bias = True),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, kernel_size = 6, stride = 2, bias = True),
                nn.ReLU(),
                nn.ConvTranspose2d(16, image_channels, kernel_size = 6, stride = 2, bias = True),
                nn.Sigmoid(),)

        self.encoder.to(self.device)
        self.fc1.to(self.device)
        self.fc2.to(self.device)
        self.fc3.to(self.device)
        self.decoder.to(self.device)

    def reparmeterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std*esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparmeterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        x_reconstruction = self.decode(z)
        return x_reconstruction, z, mu, logvar

    def predict(self, x):
        z, mu, logvar = self.encode(x)
        predict = self.decode(mu)
        return predict

    def save(self, path, step):
        torch.save(self.state_dict(), '%s/vae_%s.pt' % (path, step))

    def load(self, path, step, map_location = None):
        self.load_state_dict(torch.load("%s/vae_%s.pt" % (path, step), map_location=map_location))



if __name__ == '__main__':
    test = torch.rand(8, 3, 78, 78)
    model = VAE()
    out = model(test)
    print(out.shape)

