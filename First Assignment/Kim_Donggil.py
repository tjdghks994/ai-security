"""
CNN (Convolutional Neural Network)

- Convolution 계층과 Pooling 계층을 통해 원본 이미지에 필터링 기법을 적용하여 특징값을 추출하여 연산을 진행하는 이미지 분류에 적합한 딥러닝 모델
- AlexNet, VGG, GoogleNet, ResNet 등 다양한 CNN모델들이 있음
- 기본적인 CNN모델에 대해 알아보고 ResNet 모델에 대해 공부해보았음

"""

'''
Basic CNN

- Mnist 데이터셋을 불러와서 0~9까지의 숫자를 분류하는 기본적인 CNN모델
- 3개의 Convoutinal layer 층
- 2개의 fully connected layer 층
'''

# 1. 라이브러리 호출 및 세팅

import numpy as np
import torch
import torch.nn as nn                                       #모든 신경망 기본이 되는 패키지
import torch.optim as optim                                 #SGD, RMSProp, LBFGS, Adam 등과 같은 최적화 함수들 패키지
import torch.nn.init as init
import torchvision.datasets as dset                         #Mnist 데이터셋을 가져오고 데이터 형태를 변환시키기 위한 라이브러리들
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader     
from torch.autograd import Variable                         #자동으로 미분값 계산하기 위한 라이브러리

# 매개변수 설정
batch_size = 16                                             #한번에 얼만큼의 데이터씩 진행할 것인지
learning_rate = 0.0002                                      #학습률
num_epoch = 10                                              #몇번 학습을 진행할 것인지


# 2. 데이터셋 불러오기

mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test = dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test.__getitem__(0)[0].size(), mnist_test.__len__()
# MNIST데이터셋 훈련, 시험데이터 다운로드
# transform 함수를 통해 다운받은 데이터를 텐서형태로 변환

train_loader = torch.utils.data.DataLoader(mnist_train,batch_size=batch_size, shuffle=True,num_workers=2,drop_last=True)
test_loader = torch.utils.data.DataLoader(mnist_test,batch_size=batch_size, shuffle=False,num_workers=2,drop_last=True)
# train_loader에 훈련데이터 저장, test_loader에 시험데이터 저장
# batch_size는 위에서 설정한 16, shuffle은 데이터를 무작위로 섞을지 순서대로할지, num_workers는 데이터 로딩을 위해 몇개의 서브 프로세서를 사용할 지 정함


# 3. CNN모델 설계

class CNN(nn.Module):                  # torch.nn.Module을 상속받음
    def __init__(self):
        super(CNN,self).__init__()     # 상속받은 부모 클래스의 메소드를 사용하기 위해 super()사용
        self.layer = nn.Sequential(    # nn.Sequential은 여러 모듈을 편하게 순차적으로 실행하도록 담는 컨테이너 같은 것
            nn.Conv2d(1,16,5),         # 합성곱 계층, 정보 추출하는 과정
                                       # Conv2d의 parameters는 차례대로 in_channels, out_channels, kernel_size, stride, padding, diction , groups, bias
                                       # 필수 요소는 앞에 3개로, 인풋, 아웃풋, 커널은 필터의 크기임
                                       # MNIST의 숫자이미지는 흑백이기 때문에 인풋채널은 1, 필터는 5*5로 설정
            nn.ReLU(),                 # 활성화함수, 위 합성곱계층 통과한거 Relu 활성화함수에 넣어줌
            nn.Conv2d(16,32,5),        # 위의 과정 반복, 이 때 인풋은 앞의 층의 아웃풋채널 수와 같아야 됨
            nn.ReLU(),
            nn.MaxPool2d(2,2),         # 가장 큰 값을 뽑아내는 Max-Pooling 사용
            nn.Conv2d(32,64,5),      
            nn.ReLU(),
            nn.MaxPool2d(2,2)          # 여기까지 Convolution-Relu-Pooling 반복의 3개의 합성곱계층 설계
        )
        self.fc_layer = nn.Sequential( # fully connected layer 설계
            nn.Linear(64*3*3,100),     # Linear모듈은 선형 함수를 사용해 입력으로부터 출력을 계산하고, 내부 텐서에 가중치와 편향을 저장
                                       # 여기서 인풋 채널 설정은 위의 합성곱계층에서 맨처음 28*28의 데이터에서 별도의 패딩과정이 없기 때문에 첫번째 Conv2d의 5*5의 필터를 거쳐서 24*24, 다음 필터 거쳐서 20*20, 2*2 풀링해서 10*10
                                       # 다음 필터 겨처서 6*6, 마지막 풀링해서 3*3의 데이터가 마지막 64개의 아웃풋 채널로 나오기 때문에 64*3*3의 입력값 크기로 설정해줌
            nn.ReLU(),                 # 활성화 함수
            nn.Linear(100,10)          # 0~9까지의 숫자를 분별하는 것이 목표이기 때문에 10개의 최종 출력값 채널 설정
        )       
        
    def forward(self,x):               # Module 를 상속받은 클래스는 신경망의 정방향 계산을 수행하는 forward() 메소드를 구현해야만 함
                                       # foward 메소드는 이 모델을 호출하면 자동으로 실행 됨
        out = self.layer(x)            # 합성곱 계층 수행
        out = out.view(batch_size,-1)  # torch.view를 사용하여 텐서의 사이즈를 변경하거나 Shape을 변경할 수 있음
                                       # 사이즈를 -1로 설정하는 것은 해당 값을 유추하여 자동으로 설정함
        out = self.fc_layer(out)       # fully connected layer 계층 수행

        return out

model = CNN().cuda()                   #위 CNN클래스의 인스턴스로 model 선언


# 4. Loss&Optimizer 선택 

loss_func = nn.CrossEntropyLoss()
# 신경망의 성능 평가할 손실함수로 CrossEntropyLoss사용
# 예측 확률분포인 Q(x)와 실제 확률분포인 P(x) 둘 간의 차이를 구해 손실값을 구하는 방식
# 신경망의 결과값인 Q(x)의 값이 결과값의 합을 1로 내주는 Softmax를 통해 결과값이 확률의 의미를 가지게 함

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 
# 신경망 최적화 방법으로 확률적경사하강법 사용
# 기울기를 통해 손실함수의 최저값으로 향하도록 매개변수를 갱신시켜 최적화시키는 방법
# 학습률은 미리 설정해준 0.0002


# 5. 훈련데이터로 신경망 훈련

for i in range(num_epoch):                           #위에서 설정해준 대로 10번 반복
    for j,[image,label] in enumerate(train_loader):  #순서가 있는 자료형(리스트)을 입력으로 받아 인덱스 값을 포함하는 enumerate 객체를 리턴하는 방법
                                                     #[image,label]리스트에 MNIST 훈련 데이터의 이미지데이터,정답인 라벨데이터를 넣어줌
        x = Variable(image).cuda()                   #변수 선언, Variable로 감싸주는 것은 자동미분화 위해서
        y_= Variable(label).cuda()
        
        optimizer.zero_grad()                        #갱신할 변수들에 대한 모든 변화도를 0으로 만듬, 매번 backward()가 실행될 떄 gradient가 더해지는 구조여서 버퍼가 누적되기 때문에 초기화해줘야 됨          
        output = model.forward(x)                    #이미지 데이터를 위의 모델에 넣어주고 출력값 output
        loss = loss_func(output,y_)                  #output데이터(예측한 값)과 y_(실제 값)을 loss함수에 넣어서 성능 평가
        loss.backward()                              #역전파 단계: 모델의 매개변수에 대한 손실의 변화도를 계산함
        optimizer.step()                             #Optimizer의 step 함수를 호출하면 매개변수가 갱신됨
                                                     #위의 과정 반복하면서 매개변수 최적화됨 
        if j % 2000 == 0: 
            print(loss)                              #1000번째마다 loss값 출력          


# 6. 시험데이터로 테스트

correct = 0                                                  #correct, total 변수 초기화
total = 0

for image,label in test_loader:                              #image,label데이터를 시험데이터에서 가져옴
    x = Variable(image,volatile=True).cuda()
    y_= Variable(label).cuda()

    output = model.forward(x)                                #시험데이터의 이미지데이터를 모델에 넣어줘서 출력값 계산
    _,output_index = torch.max(output,1)                     #출력값의 가장 큰 값의 인덱스값 확인, 즉 0~9숫자 중 어떤 숫자일지 예측한 값
                                                             #max(output,1)에서 1은 한번에 1차원의 데이터 중 가장 큰 값 뽑기위함
    total += label.size(0)                                   #실제 데이터 크기만큼 total 값 더해줌
    correct += (output_index == y_).sum().float()            #예측한 데이터(output_index)와 실제 데이터(y_)가 일치하면 correct 값 추가
    
print("Accuracy of Test Data: {}".format(100*correct/total)) #정확도 계산

# batch_size = 16, epoch = 10 으로 훈련, 시험한 결과 Loss값 0.2348, 정확도 약 90% 나오는 것 확인
# https://github.com/GunhoChoi/PyTorch-FastCampus/blob/master/03_CNN_Basics/0_MNIST/3_CNN_clean.ipynb
# 해당 github 참조하여 구성

'''
ResNet

- 2015 ILSVRC 평가 1st place
- 이미지 폴더에서 개와 원펀맨의 두가지 class로 이미지 분류하는 ResNet모델 설계
- ResNet-50, 50개의 layer로 설계
- ResNet은 기존 VGG, GoogleNet 등의 CNN망들이 레이어가 깊어질수록 학습이 제대로 이루어지지 않고 오히려 에러율이 높아지는 문제 해결
- 기존의 네트워크는 단순히 forward로 weight layer을 보내는 것인 반면에, ResNet은 H(x) = F(x) + x로 Residual Connection 을 남겨주어 이전 input x를 그대로 layer의 결과와 합쳐줌으로써
Backward시 Loss에 대한 Gradient가 직접적으로 최상위 층까지 흘러갈 수 있는 길이 생기기 때문에 layer가 깊어지더라도 효과적으로 학습 가능
- Inception 모듈에서와 같이 1x1 필터 Conv를 통해 레이어가 깊어지더라도 파라미터가 많아지지 않게하여 계산량 줄임.
'''

#!/usr/bin/env python
#coding: utf-8

#필요한 라이브러리들 호출
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

#매개변수 설정
batch_size= 1
learning_rate = 0.0002
epoch = 100

#현재 디렉토리에 있는 이미지 폴더에서 이미지 불러옴,
#이미지 형태, 크기 변환
img_dir = "./images"
img_data = dset.ImageFolder(img_dir, transforms.Compose([ 
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ]))

img_batch = data.DataLoader(img_data, batch_size=batch_size,
                            shuffle=True, num_workers=2)


###Basic Block

# 1x1필터, stride = 1의 Conv, 활성화함수 실행 Block
def conv_block_1(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=1, stride=1),
        act_fn,
    )
    return model

# 1x1필터, stride = 2의 Conv, 활성화함수 실행 Block
def conv_block_1_stride_2(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=1, stride=2),
        act_fn,
    )
    return model

#1x1필터, stride = 1의 Conv, 활성화함수 실행 X Block
def conv_block_1_n(in_dim,out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=1, stride=1),
    )
    return model
  
#1x1필터, stride = 2의 Conv, 활성화함수 실행 X Block
def conv_block_1_stride_2_n(in_dim,out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=1, stride=2),
    )
    return model

#3x3필터, stride = 1의 Conv, 활성화함수 실행 Block  
def conv_block_3(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=3, stride=1, padding=1),
        act_fn,
    )
    return model


### Bottle Neck Module

# 3가지 BottleNeck 클래스 정의
# 3x3필터 앞, 뒤로 1x1필터 사용하는 이유는 인셉션 모듈과 유사, 이에 관해서는 아래 모델 설계에서 설명
# 1. 1x1필터, 3x3필터, 1x1필터 Conv지나는 BottleNeck 클래스 slef.downsample을 통해 F(x)와 x의 크기를 갖게해주고  더해줌
# 2. 1x1필터, 3x3필터, 1x1필터 Conv만 지나고 downsample 하지 않는 BottleNeck_no_down 클래스
# 3. 1x1필터, 3x3필터, 1x1필터 Conv지나고 1x1필터, stride 2로 설정하여 output size도 다운샘플링하는 BottleNeck_stride 클래스 맵의 크기와 갯수를 맞추어준 다음 더해주는 클래스, 프로젝션 숏컷

class BottleNeck(nn.Module):
    
    def __init__(self,in_dim,mid_dim,out_dim,act_fn):
        super(BottleNeck,self).__init__()
        self.layer = nn.Sequential(
            conv_block_1(in_dim,mid_dim,act_fn),     #1x1필터의 Conv와 활성화함수
            conv_block_3(mid_dim,mid_dim,act_fn),    #3x3필터의 Conv와 활성화함수
            conv_block_1_n(mid_dim,out_dim),         #1x1필터의 Conv와 활성화함수  
        )
        self.downsample = nn.Conv2d(in_dim,out_dim,1,1)   #1x1 필터로 다운샘플링
        
    def forward(self,x):                   
        downsample = self.downsample(x)
        out = self.layer(x)
        out = out + downsample                       #ResNet의 핵심이 되는 H(x) = F(x) + x 식 구현
                                                     #F(x) + x 연산을 할 때 중요한 것은 이 둘의 dimension이 같도록 모델을 설계해야 함        
        return out

    
class BottleNeck_no_down(nn.Module):
    
    def __init__(self,in_dim,mid_dim,out_dim,act_fn):
        super(BottleNeck_no_down,self).__init__()
        self.layer = nn.Sequential(
            conv_block_1(in_dim,mid_dim,act_fn),
            conv_block_3(mid_dim,mid_dim,act_fn),
            conv_block_1_n(mid_dim,out_dim),
        )
        
    def forward(self,x):
        out = self.layer(x)
        out = out + x
        
        return out

    
class BottleNeck_stride(nn.Module):
    
    def __init__(self,in_dim,mid_dim,out_dim,act_fn):
        super(BottleNeck_stride,self).__init__()
        self.layer = nn.Sequential(
            conv_block_1_stride_2(in_dim,mid_dim,act_fn),
            conv_block_3(mid_dim,mid_dim,act_fn),
            conv_block_1_n(mid_dim,out_dim),
        )
        self.downsample = nn.Conv2d(in_dim,out_dim,1,2)     #stride 2의 1x1필터 Conv를 통해 이미지의 output 사이즈도 줄임
        
    def forward(self,x):
        downsample = self.downsample(x) 
        out = self.layer(x)
        out = out + downsample
        
        return out


### ResNet Model

#ResNet 모델 설계
# 총 50-layer, Conv통과 후 average 풀링, 50layer의 매 층마다 Relu활성화 함수를 사용하기 때문에 비선형성이 높아 기존 CNN모델과 같이 여러 층의 FCL(fully connected layer)을 가지지 않음
# 대신 맵별로 평균값을 구하고 이 평균값들의 벡터를 특징값으로 사용
# layer_1에서 컬러사진이기 때문에 input_channel 3, output_channel은 아래에서 64로 입력해줌, 7x7필터, stride 2, padding 3의 Conv 통과
# layer_2에서 {1x1,3x3,1x1} x 3의 Conv 통과, 첫번째 1x1필터를 통해 3x3필터를 통과하기 전 채널 수를 작게 조절해줘서 파라미터의 수를 줄여줘 계산량 증가, 다운샘플링을 통해 맵의 사이즈 및 채널 수 조절
# layer_3에서 {1x1,3x3,1x1} x 4의 Conv 통과, 반복
# layer_4에서 {1x1,3x3,1x1} x 6의 Conv 통과, 반복
# layer_5에서 {1x1,3x3,1x1} x 3의 Conv 통과, 반복
# Avg풀링을 통해 각 맵별 평균값을 특징값으로 사용, 7x7의 맵 사이즈를 1x1로 만듬
# 마지막 Linear함수를 통해 연산, 최종 결과값 크기는 분류하려는 클래스 숫자만큼 설정

class ResNet(nn.Module):

    def __init__(self, base_dim, num_classes=2):
        super(ResNet, self).__init__()
        self.act_fn = nn.ReLU()
        self.layer_1 = nn.Sequential(                                           #입력크기 224x224x3 -> 7x7필터 -> 112x112x64(base_dim) -> 3x3 pooling -> 56x56x64
            nn.Conv2d(3,base_dim,7,2,3),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1),
        )
        self.layer_2 = nn.Sequential(                                           #입력크기 56x56x64 -> {1x1,3x3,1x1}x3 필터 -> 28x28x256
            BottleNeck(base_dim,base_dim,base_dim*4,self.act_fn),
            BottleNeck_no_down(base_dim*4,base_dim,base_dim*4,self.act_fn),
            BottleNeck_stride(base_dim*4,base_dim,base_dim*4,self.act_fn),
        )   
        self.layer_3 = nn.Sequential(                                           #입력크기 28x28x256 -> {1x1,3x3,1x1}x4 필터 -> 14x14x512
            BottleNeck(base_dim*4,base_dim*2,base_dim*8,self.act_fn),
            BottleNeck_no_down(base_dim*8,base_dim*2,base_dim*8,self.act_fn),
            BottleNeck_no_down(base_dim*8,base_dim*2,base_dim*8,self.act_fn),
            BottleNeck_stride(base_dim*8,base_dim*2,base_dim*8,self.act_fn),
        )
        self.layer_4 = nn.Sequential(                                           #입력크기 14x14x512 -> {1x1,3x3,1x1}x6 필터 -> 7x7x1024
            BottleNeck(base_dim*8,base_dim*4,base_dim*16,self.act_fn),
            BottleNeck_no_down(base_dim*16,base_dim*4,base_dim*16,self.act_fn),
            BottleNeck_no_down(base_dim*16,base_dim*4,base_dim*16,self.act_fn),            
            BottleNeck_no_down(base_dim*16,base_dim*4,base_dim*16,self.act_fn),
            BottleNeck_no_down(base_dim*16,base_dim*4,base_dim*16,self.act_fn),
            BottleNeck_stride(base_dim*16,base_dim*4,base_dim*16,self.act_fn),
        )
        self.layer_5 = nn.Sequential(                                           #입력크기 7x7x1024 -> {1x1,3x3,1x1}x3 필터 -> 7x7x2048
            BottleNeck(base_dim*16,base_dim*8,base_dim*32,nn.ReLU()),
            BottleNeck_no_down(base_dim*32,base_dim*8,base_dim*32,self.act_fn),
            BottleNeck(base_dim*32,base_dim*8,base_dim*32,self.act_fn),
        )
        self.avgpool = nn.AvgPool2d(7,1)                                        # 7x7x2048 -> 7x7 pooling -> 1x1x2048
        self.fc_layer = nn.Linear(base_dim*32,num_classes)                      # 최종 출력값은 분류하려는 클래스 숫자만큼 설정(num_classes)
       
    #위 레이어들 순차적으로 실행
    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.avgpool(out)
        out = out.view(batch_size,-1)                                           # 이미지 분류를 위해 1x1x2048의 맵을 1차원으로 펴줌
        out = self.fc_layer(out)
        
        return out
    
model = ResNet(base_dim=64).cuda()                                              # ResNet 클래스의 인스턴스로 model 선언, base_dim은 64로 설정


### Optimizer & Loss

#손실함수로 CrossEntropyLoss사용
#예측 확률분포인 Q(x)와 실제 확률분포인 P(x) 둘 간의 차이를 구해 손실값을 구하는 방식
#신경망의 결과값인 Q(x)의 값이 결과값의 합을 1로 내주는 Softmax를 통해 결과값이 확률의 의미를 가지게 함
loss_func = nn.CrossEntropyLoss()

#최적화 함수로 Adam 사용
optimizer = optim.Adam(model.parameters(),lr=learning_rate)


### Train

for i in range(epoch):
    for img,label in img_batch:
        img = Variable(img).cuda()
        label = Variable(label).cuda()

        optimizer.zero_grad()
        output = model(img)
        
        loss = loss_func(output,label)
        loss.backward()
        optimizer.step()

    if i % 10 ==0:
        print(loss)

# 구글 Colalab으로 실행해보려 했으나 구글 드라이브 같은 경로에 있는 이미지 폴더를 찾을 수 없다는 에러가 떠서 직접 돌려보지는 못함
# 다른 이미지를 다운받아 다른 경로로 설정, 다른 코드도 사용해보았으나 계속 같은 에러 발생
# https://github.com/GunhoChoi/PyTorch-FastCampus/blob/master/04_CNN_Advanced/3_ResNet.ipynb
# 해당 github 참조하여 구성
