import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time, sleep
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
from PIL import Image
import pkbar
from torch.utils.tensorboard import SummaryWriter
import Model
import torchvision.transforms as tt

def show_example(data):
    fig = plt.figure(figsize=(10, 10))
    columns = 1
    rows = 1
    for i in range(1, 2):
        img, label = data[i]
        fig.add_subplot(rows, columns, i)
        plt.title(trainset.classes[label])
        plt.imshow(img[0])
        plt.xticks([])
        plt.yticks([])

    plt.subplots_adjust(wspace=0.7)
    plt.show()





cuda = True if torch.cuda.is_available() else False
device = torch.device('cpu')
if cuda:
    device = torch.device('cuda')
torch.cuda.empty_cache()
print(cuda)




writer = SummaryWriter('runs/downsize')

transforms = torchvision.transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dir = "datasetCars"
trainset = datasets.ImageFolder(dir + '/train', transform=transforms)
testset = datasets.ImageFolder(dir + '/test', transform=transforms)


print("Total No of Images in dataset:", len(trainset) + len(testset))
print("No of images in Training dataset:    ", len(trainset))
print("No of images in Testing dataset:     ", len(testset))

l = trainset.classes
l.sort()
print("No of classes: ", len(l))
print("List of all classes")
print(l)

batch_size=32
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True)

model = Model.Net_4().to(device)

loss_fn = nn.NLLLoss().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)


epochs = 31
epoch=0
#checkpoint = torch.load('checkpoints/chk9.pth')
#model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#epoch = checkpoint['epoch'] + 1
#loss = checkpoint['loss']
train_per_epoch = int(len(trainset) / batch_size)
for e in range(epoch, epochs):
    print("")
    kbar = pkbar.Kbar(target=train_per_epoch, epoch=e, num_epochs=epochs, width=20, always_stateful=False)
    for idx, (images, labels) in enumerate(train_loader):

        images = images.to(device)

        optimizer.zero_grad()

        output = model(images)

        labels = labels.to(device)

        loss = loss_fn(output, labels)

        loss.backward()

        optimizer.step()


        writer.add_scalar('loss', loss.item(), (e * train_per_epoch) + idx)
        predictions = output.argmax(dim=1, keepdim=True).squeeze()
        correct = (predictions == labels).sum().item()
        accuracy = correct / len(predictions)
        kbar.update(idx, values=[("loss", loss), ("acc", accuracy)])
        writer.add_scalar('acc', accuracy, (e * train_per_epoch) + idx)

    if e%5==0:
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_fn
        }, 'checkpoints/chk_downsize_' + str(e) + '.pth')


num_correct = 0
num_samples = 0
model.eval()
print("\n")
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device=device)
        y = y.to(device=device)

        scores = model(x)
        _, predictions = scores.max(1)
        num_correct += (predictions == y).sum()
        num_samples += predictions.size(0)

    print(f'Dobio sam točnih {num_correct} od ukupno {num_samples} što čini točnost od {float(num_correct) / float(num_samples) * 100:.2f}%')

