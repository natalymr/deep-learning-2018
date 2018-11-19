import torch.nn as nn


class DCGenerator(nn.Module):

    def __init__(self, image_size):
        super(DCGenerator, self).__init__()

        self.layer1 = nn.ConvTranspose2d(100, image_size * 8, kernel_size=4, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(image_size * 8)
        self.activation1 = nn.ReLU()

        self.layer2 = nn.ConvTranspose2d(image_size * 8, image_size * 4, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(image_size * 4)
        self.activation2 = nn.ReLU()

        self.layer3 = nn.ConvTranspose2d(image_size * 4, image_size * 2, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(image_size * 2)
        self.activation3 = nn.ReLU()

        self.layer4 = nn.ConvTranspose2d(image_size * 2, image_size, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(image_size)
        self.activation4 = nn.ReLU6()

        self.layer5 = nn.ConvTranspose2d(image_size * 1, 3, kernel_size=4, stride=2, padding=1)
        self.activation5 = nn.Tanh()

    # weight_init
    # def weight_init(self, mean, std):
    #     for m in self._modules:
    #         normal_init(self._modules[m], mean, std)

    def forward(self, data):
        x = self.activation1(self.bn1(self.layer1(data)))
        x = self.activation2(self.bn2(self.layer2(x)))
        x = self.activation3(self.bn3(self.layer3(x)))
        x = self.activation4(self.bn4(self.layer4(x)))
        x = self.activation5(self.layer5(x))

        return x


class DCDiscriminator(nn.Module):

    def __init__(self, image_size):
        super(DCDiscriminator, self).__init__()

        self.layer1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.activation1 = nn.LeakyReLU(0.2)

        self.layer2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.activation2 = nn.LeakyReLU(0.2)

        self.layer3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.activation3 = nn.LeakyReLU(0.2)

        self.layer4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.activation4 = nn.LeakyReLU(0.2)

        self.layer5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)# nn.Linear(1024, 1)#(image_size - 6) * (image_size - 6)*3
        self.activation5 = nn.Sigmoid()
        self.image_size = image_size

    # weight_init
    # def weight_init(self, mean, std):
    #     for m in self._modules:
    #         normal_init(self._modules[m], mean, std)

    def forward(self, data):
        x = self.activation1(self.bn1(self.layer1(data)))
        x = self.activation2(self.bn2(self.layer2(x)))
        x = self.activation3(self.bn3(self.layer3(x)))
        x = self.activation4(self.bn4(self.layer4(x)))
        x = self.activation5(self.layer5(x))#.reshape(x.shape[0],
                                                 #  1024)))#(self.image_size - 6) * (self.image_size - 6)*3)))

        return x.view(-1, 1).squeeze(1)
