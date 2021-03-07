import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "./dfgcat (1)"

import argparse
import sys

# Create the parser
my_parser = argparse.ArgumentParser(description='Training')

# Add the arguments
my_parser.add_argument('model',
                       metavar='model',
                       type=str,
                       help='the model')

my_parser.add_argument('epochs',
                       metavar='epochs',
                       type=int,
                       help='number of epochs')

my_parser.add_argument('input_size',
                       metavar='input_size',
                       type=int,
                       help='input_size')
# Execute the parse_args() method
args = my_parser.parse_args()

model_name = args.model
num_epochs = args.epochs
input_size = args.input_size
if not model_name or not num_epochs or not input_size :
    print('provide model , epochs , input_size')
    print("Model name:- resnet, alexnet, vgg, squeezenet, densenet, inception")
    sys.exit()

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
# model_name = "squeezenet"

# input_size = 224

# Number of classes in the dataset
num_classes = 71

# Batch size for training (change depending on how much memory you have)
batch_size = 64

# Number of epochs to train for
# num_epochs = 15000

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

# Weights for imbalanced classes
weights = []

train_acc = []
tran_loss = []
val_loss = []

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'train' :
                train_acc.append(epoch_acc)
                tran_loss.append(epoch_loss)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    global input_size

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        # input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        # input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        # input_size = 150

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_1(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        # input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        # input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        # input_size = 299

    elif model_name == "custom" :
        model_ft = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2,inplace=True),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2,inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=34*34*3*64,out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64,out_features=num_classes),
            nn.Softmax(dim=1)
        )

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)

# Print the model we just instantiated
print(model_ft)

# Just normalization for validation

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
def load_split_train_test(datadir, valid_size = .2):
    global weights
    train_transforms = transforms.Compose([transforms.Resize((input_size,input_size)),
                                       transforms.ToTensor(),
                                    #    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                       ])
    train_data = datasets.ImageFolder(datadir,       
                    transform=train_transforms)
    num_train = len(train_data)
    imgs_per_class = []
    count = 0
    for i in range(len(train_data.imgs)):
        if i== len(train_data.imgs)-1 :
            imgs_per_class.append(count)
        elif train_data.imgs[i][1] == train_data.imgs[i+1][1] :
            count+=1
        else :
            imgs_per_class.append(count)
            count=0

    print (imgs_per_class)
    total = sum(imgs_per_class)
    for i in range(len(imgs_per_class)):
        weights.append(1.00*(total-imgs_per_class[i])/total)
    print(weights)
    indices = list(range(num_train))
    print(f"Number of total samples are {len(indices)}")
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    print(f"Number of train samples are {len(train_idx)}")
    print(f"Number of val samples are {len(test_idx)}")
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(train_data,
                   sampler=test_sampler, batch_size=batch_size)

    return trainloader, testloader
trainloader, testloader = load_split_train_test(data_dir, 0.3)

image_datasets = {'train': trainloader,'val':testloader}
# Create training and validation dataloaders
dataloaders_dict = image_datasets

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(params_to_update)

# Setup the loss fxn
# criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights,device='cuda:0'))
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))
torch.save(model_ft, 'v1.pth')
ohist = []
shist = []

ohist = [h.cpu().numpy() for h in hist]

plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(range(1,num_epochs+1),ohist,label="Pretrained")
plt.ylim((0,1.))
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.legend()
plt.show()

plt.plot(range(1,num_epochs+1),train_acc,label="Train Acc")
plt.plot(range(1,num_epochs+1),tran_loss,label="Train Loss")
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.legend()
plt.show()
