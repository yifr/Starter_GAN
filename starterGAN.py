import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
#from utils import Logger

def mnist_data():
    compose = transforms.Compose(
            [transforms.ToTensor(),
                transforms.Normalize((.5,.5,.5), (.5,.5,.5))
             ])
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

#Load data
data = mnist_data()

#Create loader so we can iterate through data
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)

#Define number of batches:
num_batches = len(data_loader)

def images_to_vectors(images):
    return images.view(images.size(0), 784)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)


#Generates 1-d vector of gaussian sampled random values:
def noise(size):
    n = Variable(torch.randn(size, 100))
    return n


class DiscriminatorNet(torch.nn.Module):
    # Three hidden-layer discriminative neural network
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 784
        n_out = 1

        self.hidden0 = nn.Sequential(
                nn.Linear(n_features, 1024),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
                nn.Linear(1024, 512),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
        )
        self.hidden3 = nn.Sequential(
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
                torch.nn.Linear(256, n_out),
                torch.nn.Sigmoid()
        )
        def forward(self, x):
            x = self.hidden0(x)
            x = self.hidden1(x)
            x = self.hidden2(x)
            x = self.out(x)
            return x

discriminator = DiscriminatorNet()

class GeneratorNet(torch.nn.Module):
    #Three hidden-layer generative network
    def __init(self):
        super(GeneratorNet, self).__init__()
        n_features = 100
        n_out = 784

        self.hidden0 = nn.Sequential(
                nn.Linear(n_features, 256),
                nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(
                nn.Linear(256, 512),
                nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
                nn.Linear(512, 1024),
                nn.LeakyReLU(0.2)
        )
        self.out = nn.Sequential(
                nn.Linear(1024, n_out),
                nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

generator = GeneratorNet()

d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

#Binary Cross Entropy Loss
#L = {l1,...,ln}^T, li = -wi[yi*log(vi) + (1-y)*log(1-v1)]
loss = nn.BCELoss()

#1D Tensor containing ones
def ones_target(size):
    data = Variable(torch.ones(size, 1))
    return data

#1D Tensor containing all zeros
def zeros_target(size):
    data = Variable(torch.zeros(size, 1))
    return data

def train_discriminator(optimizer, real_data, fake_data):
    N = real_data.size(0)
    #Reset gradients
    optimizer.zero_grad()

    #1.1 Train on real data
    predict_actual = discriminator(real_data)
    #Calculate error and backpropogate
    error_real = loss(prediction_real, ones_target(N))
    error_real.backward()

    #1.2 Train on Fake data:
    prediction_fake = discriminator(fake_data)
    #Calculate error and backpropogate
    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward()

    #1.3 Update weights with gradients
    optimizer.step()

    #Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake

def train_
