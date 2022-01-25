import torch.nn as nn

class PommerLinearAutoencoder(nn.Module):
    """
    An autoencoder containing only linear layers.
    """
    def __init__(self, num_features, mode='both'):
        """
        Initializes the autoencoder.

        :param num_features: The number of input and output features
        :param mode: The mode in which the autoencoder shall be used.
            One of 'encode' or 'decode' or 'both'.
        """
        super(PommerLinearAutoencoder, self).__init__()
        self.modes = ['encode', 'decode', 'both']
        if mode in self.modes:
            self.mode = mode
        else:
            raise Exception(f"Mode must be one of {self.modes}")

        self.encoder = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=num_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        if self.mode == 'encode':
            return self.encoder(x)

        if self.mode == 'decode':
            return self.decoder(x)

        if self.mode == 'both':
            y = self.encoder(x)
            y = self.decoder(y)
            return y