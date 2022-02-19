import numpy as np 
import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transf
import torch.nn as nn



def extr(folder_path, out_classes):

    # Hyperparameter tuning
    epoch_n = 10
    l_rate = 0.0001
    batch_size_tr = 50
    batch_size_val = 30


    # Augmentation
    transform = transf.Compose([
                transf.Resize((224,224)),
                transf.ToTensor()
    ])

    train_ds = ImageFolder(folder_path+'/train', transform=transform)
    val_ds = ImageFolder(folder_path+'/val', transform=transform)
    train_load = DataLoader(dataset=train_ds, batch_size=batch_size_tr, shuffle=True, drop_last=True)
    val_load = DataLoader(dataset=val_ds, batch_size=batch_size_val, shuffle=True, drop_last=True)
    if torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'

    
    # Model defined
    class refined(nn.Module):
        def __init__(self):
            super(refined,self).__init__()
            initial_model = torchvision.models.googlenet(pretrained=True)
            self.ref_model = nn.Sequential(*list(initial_model.children())[:-2])
            self.linear = nn.Sequential(*list(initial_model.children())[-1:])
            self.flat = nn.Flatten()
    
        def forward(self,images):
            featr = self.flat(self.ref_model(images))
            output = self.linear(featr)
            return featr,output


    # Train and Val
    def train(model, criterion, optim, epoch_n):
        best_acc=0.0
        best_featr_tr = []
        best_ver_labels_ext_tr = []
        best_featr_val = []
        best_ver_labels_ext_val = []
        for epoch in range(epoch_n):
            featr_tensor_tr = np.zeros((1,1024))
            labels_ext_tr = []
            model.train()
            running_train_loss = 0.0
            running_train_acc = 0.0
            for images,labels in train_load:
                images = images.to(device)
                labels = labels.to(device)
                with torch.set_grad_enabled(True):
                    featr,output = model(images)
                    featr_tensor_tr = np.append(featr_tensor_tr,featr.cpu().detach().numpy(), axis=0)
                    labels_ext_tr = np.append(labels_ext_tr,labels.cpu().detach().numpy(),axis=0)
                    _,pred = torch.max(output,1)
                    loss = criterion(output,labels)
                    loss.backward()
                    optim.step()
                optim.zero_grad()
                running_train_loss += loss.item()*batch_size_tr
                running_train_acc += torch.sum(pred==labels)
            running_val_loss, running_val_acc, featr_tensor_val, labels_ext_val = eval(model, criterion)
            epoch_train_loss = running_train_loss/len(train_ds)
            epoch_train_acc = running_train_acc.double()/len(train_ds)
            print("Epoch: {}".format(epoch+1))
            print('-'*10)
            print('Train Loss: {:.4f}   Train Acc: {:.4f}'.format(epoch_train_loss,epoch_train_acc))
            epoch_val_loss = running_val_loss/len(val_ds)
            epoch_val_acc = running_val_acc.double()/len(val_ds)
            print('Val Loss: {:.4f}   Val Acc: {:.4f}'.format(epoch_val_loss,epoch_val_acc))
            print('\n')
            if best_acc < epoch_val_acc:
                best_acc = epoch_val_acc
                best_featr_tr = featr_tensor_tr
                best_ver_labels_ext_tr = labels_ext_tr
                best_featr_val = featr_tensor_val
                best_ver_labels_ext_val = labels_ext_val
        print("The model with the best performance has an accuracy of :{:.4f}".format(best_acc))
        return best_featr_tr, best_ver_labels_ext_tr, best_featr_val, best_ver_labels_ext_val


    def eval(model, criterion):
        model.eval()
        featr_tensor_val = np.zeros((1,1024))
        labels_ext_val = []
        running_val_loss = 0.0
        running_val_acc = 0.0
        for images,labels in val_load:
            images = images.to(device)
            labels = labels.to(device)
            featr,output = model(images)
            featr_tensor_val = np.append(featr_tensor_val,featr.cpu().detach().numpy(), axis =0)
            labels_ext_val = np.append(labels_ext_val,labels.cpu().detach().numpy(),axis=0)
            _,pred = torch.max(output,1)
            loss = criterion(output,labels)
            running_val_loss += loss.item()*batch_size_val
            running_val_acc += torch.sum(pred==labels)
        return running_val_loss, running_val_acc, featr_tensor_val, labels_ext_val


    # model initialization 
    model = refined()
    model.linear[0] = nn.Linear(model.linear[0].in_features, out_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optim = torch.optim.Adam(model.parameters(), l_rate)


    best_featr_tr, labels_tr, best_featr_val, labels_val = train(model, criterion, optim, epoch_n)
    return best_featr_tr, labels_tr, best_featr_val, labels_val