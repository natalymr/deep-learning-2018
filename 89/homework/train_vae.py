from torchvision import transforms
import torchvision.datasets as datasets
import torch.utils.data as data
from torch.optim import Adam

import homework.vae.vae as vae
import homework.vae.trainer as vae_trainer


def main():
    # TODO your code here: start
    
    image_size = 28
    num_epochs = 20
    batch_size = 64
    log_interval = 1
    
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.ToTensor()])
    
    dataset = datasets.FashionMNIST(root='fashionMNIST', download=True,
                                    transform=transform)
    
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=4, pin_memory=True)
    
    VAE_model = vae.VAE()

    trainer = vae_trainer.Trainer(model=VAE_model, train_loader=dataloader, test_loader=dataloader,
                                  optimizer=Adam(VAE_model.parameters(), lr=0.01),
                                  loss_function=vae.loss_function)

    for epoch in range(num_epochs):

        trainer.train(epoch=epoch, log_interval=log_interval)
        trainer.test(epoch=epoch, batch_size=batch_size, log_interval=log_interval)

        trainer.plot_generated(epoch=epoch, batch_size=batch_size)

    # TODO your code here: end


if __name__ == '__main__':
    main()
