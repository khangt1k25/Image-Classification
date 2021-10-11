import torch 
from utils import get_model, get_otimizer, get_transform, get_dataset_dataloader, get_xent_criterion



transform = get_transform()
trainset, trainloader = get_dataset_dataloader(split='train', transform=transform)
testset, testloader = get_dataset_dataloader(split='test', transform=transform)


model = get_model()
optimizer = get_otimizer(params=model.parameters(), lr=0.001, momentum=0.99, type='adam')
criterion = get_xent_criterion()


epochs = 10

for epoch in range(epochs):

    running_loss = 0.0
    for i, data in enumerate(trainloader):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # print every 2000 mini-batches  : Need to change to tensorboard plot and save the best model
        if i % 2000 == 1999:    
            print('[%d, %5d] loss: %.3f' % 
                    (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0 
        

print('Finished Training')
