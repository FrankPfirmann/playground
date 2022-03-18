import torch
import torch.nn as nn

class PommerConvAutoencoder(nn.Module):
    """
    An autoencoder containing two-dimensional convolutional layers.
    """
    def __init__(self, embedding_dims=128, mode='both'):
        """
        Initializes the autoencoder.

        :param num_features: The number of input and output features
        :param mode: The mode in which the autoencoder shall be used.
            One of 'encode' or 'both'.
        """
        super(PommerConvAutoencoder, self).__init__()
        self.modes = ['encode', 'both']
        self.embedding_dims=embedding_dims
        if mode in self.modes:
            self.mode = mode
        else:
            raise Exception(f"Mode must be one of {self.modes}")

        self.encoder1 = nn.Conv2d(13, 39, 3, padding=1)
        self.pool = nn.MaxPool2d(3, 3, return_indices=True)
        self.encoder2 = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=self.embedding_dims)
        )
        self.decoder1 = nn.Sequential (
            nn.Linear(in_features=embedding_dims, out_features=351),
            nn.Unflatten(1, (39, 3, 3))
        )
        self.unpool = nn.MaxUnpool2d(3, 3)
        self.decoder2 = nn.ConvTranspose2d(39, 13, 3, padding=1)

    def forward(self, x):
        x = torch.relu(self.encoder1(x))
        x, indices = self.pool(x)
        x = torch.relu(x)
        x = torch.sigmoid(self.encoder2(x))

        if self.mode == 'encode':
            return x

        x = torch.relu(self.decoder1(x))
        x = torch.relu(self.unpool(x, indices))
        x = torch.sigmoid(self.decoder2(x))

        if self.mode == 'both':
            return x