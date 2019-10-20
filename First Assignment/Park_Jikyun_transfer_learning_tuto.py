"""
전이 학습(Transfer Learning)
==================
개미와 벌을 분류(Classification)하는 문제를 통해 전이 학습을 배워본다.
전이학습이란? 이미 잘 훈련된 모델이 존재하고, 해당 모델과 유사한 문제를 해결할 때 사용

목표 : 개미 또는 벌 사진을 컴퓨터에 입력했을 때 개미인지 벌인지를 모델이 예측

CS231N 강의노트에 따르면, CNN(ConvNet, Convolutional Neural Network)을 실제 데이터풀에서 
곧바로 학습시키긴 여건상 어려우므로 ImageNet 같이 커다란 데이터셋에서 
선학습(Pre-training)을 한 다음, 이를 특징 추출기로 고정해놓고 사용하는 것이 일반적이다.

개미와 벌 각각 124, 121개의 이미지를 이용하여 전이 학습

튜토리얼 시나리오 순서
------------------
## 1. Training 및 Test 데이터셋을 불러오고 정규화 - Torchvision 사용
## 2. 예측 모델 정의 (선학습이 된 모델)
## 3. 손실함수(Loss Function)와 Optimizer(최적화 설계) 정의 
- cf) 손실함수란? - 신경망 성능의 '나쁨'을 나타내는 지표, 평균 제곱 오차 / 교차 엔트로피 오차 

## 4. Training 데이터를 사용하여 신경망을 학습
## 5. Test 데이터를 사용하여 신경망이 잘 훈련되었는지를 검사

"""

# 라이선스 : BSD(Berkeley Software Distribution) - 공공 목적, 라이선스 및 저작권 표시 조건 외에 아무런 제약X
# 저자 : Sasnk Chilamkurthy

#######
# 1. Training 및 Test 데이터셋을 불러오고 정규화하기
#######

# PyTorch, Torchvision 등 패키지를 이용해 데이터셋 불러오기
from __future__ import print_function, division # from __future__ import print_function - Py 2에서 Py 3 문법을 사용할때 쓰는 구문

import torch # PyTorch 
import torch.nn as nn # mini batch 만을 지원하는 nn 패키지
import torch.optim as optim # Optimizer 알고리즘 구현체(implementation) 제공
from torch.optim import lr_scheduler 
import numpy as np # Numpy
import torchvision 
from torchvision import datasets, models, transforms # Torchvision 이용
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   # 대화형 모드 - 수시로 작업물을 백엔드에 띄워놔서 수정가능한 모드 


# 학습을 위해 데이터 증가(augmentation) 및 정규화(normalization)
# 검증을 위한 정규화
# 학습(train), 검사(val)
data_transforms = {
    'train': transforms.Compose([ # Compose는 여러 transforms 묶는 역할 (단계적으로 진행)
        transforms.RandomResizedCrop(224), # 임의 이미지를 (224, 224) 로 Resized한 결과 추출
        transforms.RandomHorizontalFlip(), # 임의 각도 변경(fliping)
        transforms.ToTensor(), # 입력 이미지(Input Image)를 Tensor로 Converting
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 입력 데이터 크기값들, 반드시 미리 입력되어 있어야함
    ]),
    'val': transforms.Compose([
        transforms.Resize(256), # (256, 256) 으로 Resized
        transforms.CenterCrop(224), # 중심을 기준으로 (224, 224) 크기로 크롭
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'hymenoptera_data' # 데이터셋을 불러올 디렉토리 경로
path = {x: os.path.join(os.path.dirname(os.path.abspath(__file__)), data_dir, x)
                  for x in ['train', 'val']}    

# os.path.dirname : 자동으로 파일 경로를 가져오는 함수
# 불러올 파일 경로를 절대경로화(abspath) 하여 각각의 변수에 경로를 저장
# path['train'] 은 train set의 경로 / path['val'] 은 val set의 경로
# os.path.join은 문자열을 이어 붙여주는 함수

image_datasets = {x: datasets.ImageFolder(path[x], data_transforms[x])
                    for x in ['train', 'val']}  # 가져온 데이터를 image_datasets['train', 'val'] 이란 이미지 데이터셋으로 지정

dataloaders = { 'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size=4, shuffle=True, num_workers=0),
                'val' : torch.utils.data.DataLoader(image_datasets['val'], batch_size=4, shuffle=True, num_workers=0) }

# torch.utils.data.DataLoader : 불러온 데이터를 네트워크 입력에 사용하기 위해 dataset인 ImageFolder 함수, batch size, shuffle 여부 등의 인자들을 정리하는 함수
# num_workers 는 스레드 개수 설정 (0개)

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# dataset_sizes['train'] : train 데이터셋 사진 개수
# dataset_sizes['val'] : val 데이터셋 사진 개수

class_names = image_datasets['train'].classes
# class_names : ['ants', 'bees'] / 개미와 벌 클래스로 분류

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# GPU가 이용가능한지를 확인

# 일부 이미지 시각화 - 데이터 로드 정상 작동 여부 확인
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # 이미지를 확인하려면 input 값을 3초 이상으로

# 한 개의 batch 만큼 이미지를 불러온다. batch size를 4로 했으니 사진 4장이 로드됨
inputs, classes = next(iter(dataloaders['train']))
 
out = torchvision.utils.make_grid(inputs) # 로드된 데이터에 make_grid 함수를 통해 그리드 추가


imshow(out, title=[class_names[x] for x in classes]) # 이미지 출력 시도


#######
# 2. 예측 모델 정의 
#######

# Pre-trained 모델인 ResNet18 모델 불러오고 마지막 층 재정의
model_ft = models.resnet18(pretrained=True) 
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

# ResNet18 이란? 수백만 이상의 이미지 데이터베이스를 기반으로 학습된 CNN 이다. 
# ResNet18 신경망은 18개의 레이어로 이루어져 있으며, 키보드, 마우스, 동물들 등을 포함한 1000 가지 종류의 이미지를 구별할 수 있다.
# ResNet18은 (224, 224) 크기의 이미지를 입력해야 하며, 이 모델을 통해 개미와 벌을 분류하는 문제 상황에 대한 전이학습이 가능한 것이다.

# model_ft.fc.in_features : Resnet18 모델의 마지막 layer 에서 출력 노드(개미 or 벌)의 개수를 구하는 함수 // 결과값 = 2(개미, 벌) 
# 예측 모델 : ResNet18 모델의 마지막 단에 출력 노드 개수가 2인 nn.Linear layer(fully connected layer)를 추가한 모델

#######
# 3. 손실함수와 Optimizer 정의
#######

criterion = nn.CrossEntropyLoss() # 교차엔트로피오차(Cross Entropy Loss)를 계산하는 함수 

# 모든 매개변수들이 최적화되었는지 관찰
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9) 

# SGD(Stochastic Gradient Descent) 란? 매 epoch(에폭)마다 모든 데이터를 input 받아 신경망을 적용하여 오차를 줄이는 GD(경사 하강 알고리즘)와 달리,
# 전체 데이터를 batch 단위로 나누어 만든 subset(mini batch)을 이용하여 학습을 여러번에 걸치기 때문에 작업 효율성(시간 & 메모리) 측면에서 경제적인 Optimizer 이다. 
# 실제 Accuracy 도 비교해보면 SGD 방식이 GD 방식에 비해 크게 뒤처지지 않는 것으로 평가받는다.
# cf) BGD(배치 경사 하강 알고리즘) 은 전체 데이터의 gradient의 평균을 내서 오차를 수정하는 방식, Accuracy는 높지만 속도가 느림
# cf) epoch : 오차를 줄이기 위해 학습하는 step 

# 7 epoch마다 0.1씩 학습율 감소
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

#######
# 4+5. Training 데이터를 사용하여 신경망 학습 + Test 데이터를 사용하여 신경망이 잘 훈련되었는지 검사
#######

# scheduler : torch.optim.lr_scheduler 의 LR 스케줄러의 Object
# Training과 Validation을 각 epoch마다 순차적으로 진행 
def train_model(model, criterion, optimizer, scheduler, num_epochs=25): # 총 25 에폭
    since = time.time() # 시작 시간을 기록(총 소요 시간 계산을 위해)

    best_model_wts = copy.deepcopy(model.state_dict()) # 예측 모델 deepcopy(깊은 복사)
    best_acc = 0.0

    for epoch in range(num_epochs): # 매 에폭마다
        print('Epoch {}/{}'.format(epoch, num_epochs - 1)) # Epoch 1/19, Epoch 2/19 ....
        print('-' * 10)

        # 각 epoch 마다 학습 단계와 검증 단계
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step() 
                model.train()  # 모델을 학습 모드로 설정
            else:
                model.eval()   # 모델을 평가 모드로 설정

            running_loss = 0.0 # 초기화
            running_corrects = 0 # 초기화

            # 데이터를 반복
            for inputs, labels in dataloaders[phase]: # dataloader로부터 input dataset과 그에 할당된 label을 불러옴
                inputs = inputs.to(device) # GPU에 input dataset을 올림
                labels = labels.to(device) # GPU에 label을 올림

                # 매개변수 경사도를 0으로 설정
                optimizer.zero_grad()

                # 학습 시에만 연산 기록을 추적
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) # 마지막 layer에서 가장 값이 큰 1개의 class(개미 or 벌)를 예측값으로 지정 
                    loss = criterion(outputs, labels)

                    # 학습 단계인 경우 역전파 + 최적화
                    if phase == 'train': # Training 모드에서는 weight를 update
                        loss.backward()
                        optimizer.step()

                # 통계
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase] 
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)) # 교차엔트로피오차, 정확도 출력

            # 모델을 깊은 복사(deep copy)함
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 총 훈련 소요 시간
    print('Best val Acc: {:4f}'.format(best_acc)) # 이미지 분류 정확도 

    # 가장 나은 모델 가중치를 불러옴
    model.load_state_dict(best_model_wts)
    return model

print('=========modeling finished=========')

# 모델 훈련 시작
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)