from torch import nn
import torch 
from torchvision import transforms 
import torchvision
from torch.utils.data import DataLoader, random_split
from torch import optim 
from collections import defaultdict
import pickle 
from matplotlib import pyplot as plt 
import time
from Model.models import *


class Trainer:
    def __init__(self, model_name: str,
                batch_size: int = 64,
                lr: float = 1e-5,
                path_save_model: str = "",
                n_epochs: int = 100,
                use_pretrained: float = False,
                training_data_percent:float = 0.2
        ):
        self.model_name = model_name
        self.n_classes = 10
        self.batch_size = batch_size
        self.lr = lr 
        self.use_pretrained = use_pretrained
        self.training_data_percent = training_data_percent
        self.path_save_model = path_save_model
        self.n_epochs = n_epochs 
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.get_model()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.acc_test_curret = 0
        self.make_loader()
    
    def get_model(self):
        if self.model_name == "lenet":
            model = Lenet(n_classes=self.n_classes).to(self.device)
            self.input_size = 32

        elif self.model_name =="alexnet":
            model = Alexnet(n_classes=self.n_classes, use_pretrained = self.use_pretrained).to(self.device)
            self.input_size = 224
        else:
            raise Exception("Not implement")

        return model
    
    def make_loader(self):
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.train_set = torchvision.datasets.CIFAR10(root='./data', 
                                            train=True, 
                                            download=True,
                                            transform=self.transform
                                        )
        self.test_set = torchvision.datasets.CIFAR10(root='./data',
                                        train=False,
                                        download=True,
                                        transform=self.transform
                                        )
        self.n_train_samples = int(self.training_data_percent * len(self.train_set))
        self.n_val_samples = len(self.train_set) - self.n_train_samples
        self.train_set, self.val_set = random_split(
            self.train_set,
            lengths=[self.n_train_samples, self.n_val_samples],
            generator=torch.Generator().manual_seed(42),
        )
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,num_workers=2)
        self.val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True,num_workers=2)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=2)


    def train_on_epoch(self):
        self.model.train()
        train_loss = 0
        correct, total = 0, 0
        for idx,(images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            outs = self.model(images)
            self.optimizer.zero_grad()
            loss = self.loss_fn(outs, labels)
            loss.backward()
            self.optimizer.step()
            
            predict = torch.argmax(outs, dim=1)
            total += predict.size(0)
            correct += (predict == labels).sum().item()
            train_loss += loss.item()
            
            if idx % 200 == 0:
                print(idx, end=" ")
        print()
        return train_loss/len(self.train_loader), correct / total
    
    def plot(self, title, file_name, train_values, val_values,is_acc=False):
        plt.plot(train_values)
        plt.plot(val_values)
        plt.title(title)
        if is_acc:
            plt.ylabel("Accuracy")
        else:
            plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(['Train', 'Val'])
        plt.show()
        plt.savefig(file_name)

    def plot_history(self):
        self.plot('Accuracy', 'acc.png', self.history['train_acc'], self.history['val_acc'])
        self.plot('Loss', 'loss.png', self.history['train_loss'], self.history['val_loss'])

    def save_model(self, epoch):
        model_state ={
            "model" : self.model.state_dict(),
            "optimizer" : self.optimizer
        }
        torch.save(model_state, self.path_save_model+f'model{epoch}.pth')
        print("Save model done")
        with open(f"history{epoch}.pkl",'wb') as file:
            pickle.dump(self.history, file, protocol=pickle.HIGHEST_PROTOCOL)
        print("Save history done")
    
    def testing(self):
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outs = self.model(images)
                
                predict = torch.argmax(outs, dim=1)
                total += predict.size(0)
                correct += (predict == labels).sum().item()
        
        self.acc_test_curret = 100 * correct / total
        print("Accuracy for testing: {:.2f}\n".format(self.acc_test_current))

    def evaluate(self):
        val_loss = 0
        self.model.eval()
        correct, total = 0, 0
        for idx,(images, labels) in enumerate(self.val_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            outs = self.model(images)
            loss = self.loss_fn(outs, labels)
            val_loss += loss.item()
            
            predict = torch.argmax(outs, dim=1)
            total += predict.size(0)
            correct += (predict == labels).sum().item()
            if idx % 200 == 0:
                print(idx, end=" ")
        print()
        return val_loss/len(self.val_loader), correct / total
    

    def fit(self):
        self.history = defaultdict(list)
        best_acc = 0
        for epoch in range(self.n_epochs):
            start_time = time.time()
            train_loss, train_acc = self.train_on_epoch()
            val_loss, val_acc = self.evaluate()
            self.history['train_acc'] = train_acc
            self.history['train_loss'] = train_loss
            self.history['val_acc'] = val_acc
            self.history['val_loss'] = val_loss
            print(f"Epoch:{epoch}---Train acc:{train_acc}---Train loss:{train_loss}---Val acc:{val_acc}---Val loss:{val_loss}--Time:{time.time()-start_time}")            
            self.testing()
            if self.acc_test_current > best_acc and epoch >= 10:
                best_acc = self.acc_test_current 
                self.save_model(epoch)
                
