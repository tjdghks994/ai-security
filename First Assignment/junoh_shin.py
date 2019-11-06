# torchvision은 이미지/영상 처리에 특화 된 모듈들을 제공하는 패키지이다. 
# torchvision과 torchvision이 제공하는 데이터셋, 이미지 분류 및 정규화를 위한 변환기를 import한다.
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable				# 역전파 및 자동 미분을 위한 autograd

from torch import optim							# 모델 최적화 함수(SGD, adagrad, momentum, Adam 등)을 포함
import torch.nn as nn							# 신경망 구성을 위한 모듈
import torch.nn.functional as F					# 활성화 함수(Sigmoid, ReLU 등)을 포함한다. 
from torch.optim import lr_scheduler			# 일정 주기의 epoch마다 학습률을 조정하는 스케줄러
import torchvision.utils

# numpy, pandas, matplotlib은 데이터 처리를 위한 패키지이다. 
# 또한 PIL은 Python Image Library의 약자로, 이미지 처리를 위해 import한다. 
import numpy as np
import pandas as pd 
import time
import copy
import os

import matplotlib.pyplot as plt
import PIL.ImageOps  
from PIL import Image
import torch

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()
	
batch_size = 32	# 1epoch을 몇 batch씩 몇 iteration 돌릴지 결정한다. 
epochs = 20		# 몇 epoch 학습할지 결정한다. 

# 학습 데이터와 테스트 데이터의 경로와 파일명을 결정한다. 
training_dir="../input/sign_data/sign_data/train"
training_csv="../input/sign_data/sign_data/train_data.csv"
testing_csv="../input/sign_data/sign_data/test_data.csv"
testing_dir="../input/sign_data/sign_data/test"

# 데이터셋 클래스
# 아래부터 SND라고 줄여 부르기로 한다. 
class SiameseNetworkDataset():
    
    def __init__(self,training_csv=None,training_dir=None,transform=None):
        # used to prepare the labels and images path
		# 이니셜라이저 : 레이블링과 이미지 경로를 위한 준비
        self.training_df=pd.read_csv(training_csv)				#학습 파일을 읽어온다. 
        self.training_df.columns =["image1","image2","label"]	#학습 파일을 분류한다. 
        self.training_dir = training_dir    					#학습 파일의 경로
        self.transform = transform
	
	# index번째 데이터를 리턴한다. 
    def __getitem__(self,index):
        
        # getting the image path
		# 이미지 경로를 불러온다. 
        image1_path=os.path.join(self.training_dir,self.training_df.iat[index,0])
        image2_path=os.path.join(self.training_dir,self.training_df.iat[index,1])
        
        
        # Loading the image
		# 위에서 지정한 이미지 경로에서 이미지를 불러온다. 
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
		# 이미지를 256단계 명암값을 가지는 흑백 이미지로 변환한다.
        img0 = img0.convert("L") 
        img1 = img1.convert("L")
        
        # Apply image transformations
		# 트랜스폼된 이미지와 텐서 값을 리턴한다. 
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        return img0, img1 , torch.from_numpy(np.array([int(self.training_df.iat[index,2])],dtype=np.float32))
    
	#학습 자료의 개수를 리턴한다. 
    def __len__(self):
        return len(self.training_df)
		
# 설정한 경로에 있는 학습 데이터와 transforms를 통해 resize 후 구성한 tensor를 인자로 갖는 SND 클래스를 생성한다. 
train_dataset = SiameseNetworkDataset(training_csv,training_dir,transform=transforms.Compose([transforms.Resize((105,105)),transforms.ToTensor()]))

# dataloader는 배치 사이즈에 맞게 데이터를 로드시켜준다. 
train_dataloader = DataLoader(train_dataset,shuffle=True, num_workers=8,batch_size=batch_size)

# 테스트 데이터셋도 똑같이 클래스를 생성해준다. 
test_dataset = SiameseNetworkDataset(testing_csv,testing_dir,transform=transforms.Compose([transforms.Resize((105,105)),transforms.ToTensor()]))

# 테스트 데이터니까 한개만 로드시켜준다. 
test_dataloader = DataLoader(test_dataset,num_workers=6,batch_size=1,shuffle=True)

# 신경망 클래스
# nn.Module을 상속한다. 
class SiameseNetwork(nn.Module):
    def __init__(self):
		# 이니셜라이저
        super(SiameseNetwork, self).__init__()
		# 부모 클래스의 이니셜라이저를 그대로 가져온다. 
        
		# cnn을 구성해준다. 
        self.cnn1 = nn.Sequential(
			# 흑백 이미지이므로 in_channel은 1, output_channel은 96개, 필터의 사이즈는 11*11, 데이터를 건너뛰지 않기 위해 보폭은 1로 정한다. 
            nn.Conv2d(1, 96, kernel_size=11,stride=1),
			# 활성화 함수는 ReLU 함수를 사용한다. 
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),
            
			# in_channel은 이전 output인 96, output_channel은 256개, 필터의 사이즈는 5*5, 데이터를 건너뛰지 않기 위해 보폭은 1로 하고, 가장자리를 0으로 채우는 padding은 끝의 2줄로 정한다.
            nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),
			# 과적합을 방지하기 위한 드롭아웃을 0.3으로 정하여 30%를 학습에서 배제한다. 
            nn.Dropout2d(p=0.3),
			
			# in_channel은 이전 output인 256, output_channel은 384개, 필터의 사이즈는 3*3, 데이터를 건너뛰지 않기 위해 보폭은 1로 하고, padding은 끝의 1줄로 정한다.
            nn.Conv2d(256,384 , kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
			# in_channel은 이전 output인 384, output_channel은 256개, 필터의 사이즈는 3*3, 데이터를 건너뛰지 않기 위해 보폭은 1로 하고, padding은 끝의 1줄로 정한다.
            nn.Conv2d(384,256 , kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
			# 드롭아웃을 0.3으로 정하여 30%를 학습에서 배제한다.
            nn.Dropout2d(p=0.3))
        
        # Defining the fully connected layers
		# 결과값을 도출하기 위해 마지막 레이어는 FNN으로 하고 Linear nn을 구성한다. 
        self.fc1 = nn.Sequential(
            nn.Linear(30976, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),         
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128,2))
  
    def forward_once(self, x):
        # Forward pass 
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        
        return output1, output2
		
class ContrastiveLoss(torch.nn.Module):
    
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0),2))

        return loss_contrastive
		
net = SiameseNetwork().cuda()
criterion = ContrastiveLoss()
optimizer = optim.RMSprop(net.parameters(), lr=1e-4, alpha=0.99, eps=1e-8, weight_decay=0.0005, momentum=0.9)

def train():
    counter = []
    loss_history = [] 
    iteration_number= 0
    
    for epoch in range(0,epochs):
        for i, data in enumerate(train_dataloader,0):
            img0, img1 , label = data
            img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
            optimizer.zero_grad()
            output1,output2 = net(img0,img1)
            loss_contrastive = criterion(output1,output2,label)
            loss_contrastive.backward()
            optimizer.step()
            if i %50 == 0 :
                print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
                iteration_number +=10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
    return net
	
model = train()
torch.save(model.state_dict(), "model.pt")

# Load the saved model
device = torch.device('cuda')
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load("model.pt"))

# Print the sample outputs to view its dissimilarity
counter=0
list_0 = torch.FloatTensor([[0]])
list_1 = torch.FloatTensor([[1]])
for i, data in enumerate(test_dataloader,0): 
    x0, x1 , label = data
    concatenated = torch.cat((x0,x1),0)
    output1,output2 = model(x0.to(device),x1.to(device))
    eucledian_distance = F.pairwise_distance(output1, output2)
    if label==list_0:
        label="Orginial"
    else:
        label="Forged"
    
    imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f} Label: {}'.format(eucledian_distance.item(),label))
    counter=counter+1
    if counter ==20:
        break