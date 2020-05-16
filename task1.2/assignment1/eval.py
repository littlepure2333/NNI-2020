import torch
import torch.nn as nn
import torch.optim as optim

from data import testloader
from net import Net


def main():
    # load data
    # testloader = testloader

    # assign device (gpu or cpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define net
    net = Net().to(device)

    # load model
    PATH = './cifar_net.pth'
    net.load_state_dict(torch.load(PATH))

    # evaluate
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

if __name__ == "__main__":
    main()