# -*- coding: utf-8 -*-
"""Advanced CNN.ipynb

Lee_Wooseong_Advanced_reference.md 에서 이어지는 내용입니다. 
레퍼런스 파일을 참고하고 읽어주세요.

"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torchvision import datasets, transforms
# 데이터를 원활하게 가져오고 변형하기 위해 torchvision을 import
# transforms는 선처리를 위한 함수


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
batch_size = 64
# deep learning에서 분석에 사용하는 데이터는 양이 너무 방대하기 때문에 전체 데이터를 한 번의 연산으로 돌리기에는 cpu의 무리가 따름
# 때문에 전체 데이터를 몇 번에 나눠서 연산할지 결정하는 batch_size 설정
# batch_size가 클수록 연산 속도가 오래 걸림

train_dataset = datasets.MNIST(root='./mnist_data/',train = True, transform = transforms.ToTensor(),download = True) 
test_dataset = datasets.MNIST(root='./mnist_data/',train = False, transform = transforms.ToTensor()) 
# google colab의 ./data 디렉토리에 MNIST의 train과 test 데이터 다운로드
# transform 함수를 통해 다운받은 데이터를 Tensor형태로 저장함

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True) 
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)
# DataLoader에 train_dataset과 test_dataset을 가져와서 각각 train_loader와 test_loader를 설정
# batch_size는 위에서 설정한 64의 값을 그대로 가져옴
# shuffle은 데이터 순서를 shuffle은 데이터 순서를 무작위로 섞을 것인지 결정
# shuffle값이 True면 순서를 무작위로, False면 순서대로 배치

class InceptionModule(nn.Module):
# class가 곧 생성할 Neural network
# class의 이름이 네트워크의 이름
# InceptionModule이란 하나의 이미지를 다양한 사이즈의 필터를 사용해서 특징을 추출한 conv layer들을 하나로 합쳐 max pooling하는 하나의 싸이클


                                                    # 초기화 함수
   def __init__(self,in_channels):
      super(InceptionModule,self).__init__()
      # super()에 생성할 네트워크를 저장

      self.branch1x1 = nn.Conv2d(in_channels,16,kernel_size=1) 
      # 1. in_channel의 경우 사진 데이터를 통해 입력받음
      # 2. out_channel의 수, 즉 뽑아낼 conv layer들을 16겹으로 출력
      # 3. kernel_size가 1이므로 사용할 필터의 크기가 1x1
      # 1x1 필터로 먼저 conv layer를 생성한 후 이 conv layer에 대하여 연산을 진행하면 1x1 필터를 사용하지 않을 때보다 연산을 효과적으로 줄일 수 있음

      self.branch5x5_1 = nn.Conv2d(in_channels,16,kernel_size=1) 
      self.branch5x5_2 = nn.Conv2d(16,24,kernel_size=5,padding=2)
      # 1. branch1x1과 같이 크기가 1x1인 필터를 사용하여 원래 이미지에 대한 conv layer 생성
      # 2. 1x1로 출력한 conv layer를 5x5 필터를 사용하고 24겹의 out_channel로 출력.  
      # 3. out_channels의 크기가 같아야 마지막에 모든 conv layer들을 합쳐서 사용할 수 있음. 때문에 out_channel의 사이즈를 같게 하기 위해 패딩 추가

      self.branch3x3_1 = nn.Conv2d(in_channels,16,kernel_size=1) 
      self.branch3x3_2 = nn.Conv2d(16,24,kernel_size=3,padding=1) 
      self.branch3x3_3 = nn.Conv2d(24,24,kernel_size=3,padding=1)
      # 위의 branch5x5와 같이 out_channel의 사이즈를 동일하게 conv layer 출력

      self.branch_pool = nn.Conv2d(in_channels,24,kernel_size=1) 
      # average pool 이후에 1x1 필터를 사용하는 conv_layer 출력

                                                    # forward 함수 생성
   def forward(self,x):
      branch1x1 = self.branch1x1(x)
      # __init__함수에서 초기화했던 self.branch1x1에 x를 대입하여 branch1x1에 저장

      branch5x5 = self.branch5x5_1(x) 
      branch5x5 = self.branch5x5_2(branch5x5)
      # __init__함수에서 초기화했던 self.branch5x5_1()에 x를 대입한 값을 branch5x5에 저장
      # self.branch5x5_1(x)의 출력값을 다시 self.branch5x5_2에 대입하여 brach5x5에 저장

      branch3x3 = self.branch3x3_1(x) 
      branch3x3 = self.branch3x3_2(branch3x3) 
      branch3x3 = self.branch3x3_3(branch3x3)
      # brach5x5와 같은 방법으로 conv2d 과정을 모두 거친 self.branch3x3_3(branch3x3)의 값을 branch3x3에 저장

      branch_pool = F.avg_pool2d(x,kernel_size=3,stride=1,padding=1)
      branch_pool = self.branch_pool(branch_pool) 
      # 입력 이미지에 대하여 3x3 필터를 이용하고 stride=1, padding=1을 설정하여 average_pool2d를 연산한 값을 branch_pool에 저장
      # brach_pool에 저장된 값을 __init__에서 정의한 self.branch_pool()에 대입하여 branch_pool에 저장

      outputs = [branch1x1,branch5x5,branch3x3,branch_pool]
      # 위의 연산에서 구한 1x1, 5x5, 3x3, brach_pool의 conv layer들을 outputs에 하나로 통합(concatenate)
      return torch.cat(outputs,1)

                          # MainNet 설정
                          # MainNet에서 앞서 설정한 Inception module들을 연결해주는 역할을 함
class MainNet(nn.Module):
     def __init__(self): 
       super(MainNet,self).__init__() 
       self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        # MNIST는 흑백 사진이기 때문에 in_channel은 1, out_channel을 10으로, 필터의 크기는 5x5로 설정
       self.conv2 = nn.Conv2d(88,20,kernel_size=5) 
        # self.conv2는 conv1에서 연산한 결과를 입력으로 받음
        # Inception Module의 forward 함수에서 도출한 outputs의 크기가 88이기 때문에(16+24+24+24) in_channel을 88로 설정, 

       self.incept1 = InceptionModule(in_channels=10) 
      # 위에서 설정한 InceptionModule 클래스에 in_channels를 10으로 설정하여 실행
       self.incept2 = InceptionModule(in_channels=20) 
      # InceptionModule 클래스에 in_channels를 20으로 설정하여 실행

       self.max_pool = nn.MaxPool2d(2)
      # max_pooling의 기본 설정값인 2x2로 연산한 activation map을 max_pooling 
       self.fc = nn.Linear(1408,10)
      # 전체 프로그램을 실행하면 나오는 에러를 통해서 최종 연산의 차원인 1408을 확인한 후 대입하여 fully connect의 input으로 넘겨줌

     def forward(self,x):
      in_size = x.size(0) 
      # size(0)을 하면 (n, 28*28)중 n을 리턴함 
      x = F.relu(self.max_pool(self.conv1(x)))
      # 위에서 정의한 conv1에 x를 대입하고 max_pooling을 통해 특징을 뽑아낸 데이터를 ReLu를 통해 activation
      # Sigmoid 함수의 경우 결과가 0~1 사이에서 값이 머무름.
      # 때문에 neural net의 hidden layer 수가 많을 때, 역전파 과정에서 weight값이 원래의 입력에 미치는 영향이 거의 미미해지는 vanishing gradients현상이 발생
      # 때문에 학습 layer의 수가 많을 때는 Sigmoid보다 ReLu 함수가 비교적 효율적임

      x = self.incept1(x)
      # conv1을 ReLu를 통해 activation한 결과를 위에서 정의한 self.incept1, 즉 Inception Module에 대입 
      x = F.relu(self.max_pool(self.conv2(x)))
      # 위에서 정의한 self.conv2에 앞서 연산한 x를 대입한 후 __init__에서 정의한 대로 2x2 크기로 Max Pooling 실행하여 x에 저장
      x = self.incept2(x) 
      # 마지막으로 self.incept2()에 위에서 과정을 거친 x 대입
      x = x.view(in_size,-1)
      # in_channel의 값만 그대로 두고 나머지 값들은 -1로 펴버림
      # batch_size * channel * width * height였다면
      # batch_size, (channel * width * height)의 형태로
      x = self.fc(x)
      # conv2d 과정과 max_pooling을 마친 1408개의 cell들에 대하여 fully connected 실행
      return F.log_softmax(x)
      # 연산된 값들을 softmax activation을 통하여 출력

model = MainNet().to(device)
# 연산에 GPU를 사용하기 위한 과정
optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.5)
# 스토케스틱 경사 하강법을 사용할 때 파라미터 값에 꼭 사용하는 model의 parameters()를 입력해줘야 함
# 학습률을 0.01로 설정하여 train 과정의 역전파에서 사용.
# 수정할 때의 momentum은 0.5로 설정
criterion = nn.NLLLoss()
# forward의 return값으로 연산을 마친 fully connected layer에 softmax 함수를 적용했으므로 NLLLoss()를 통해 라벨 분류 마무리.

# train 과정
def train(epoch): 
  model.train() 
  for batch_idx, (data, target) in enumerate(train_loader):
  # 프로그램 처음에 설정했던 train_loader에서 데이터를 가져옴
     data = data.to(device) 
     target = target.to(device) 
     
     output = model(data) 
     optimizer.zero_grad() 
     #네트워크 시작할 때 네트워크의 각 파라미터의 gradient를 0으로 초기화해줌
     loss = criterion(output, target) 
     loss.backward()
     # 계산된 loss값, 즉 원래의 라벨과 예상값의 차이에서 발생하는 loss들에 대한 역전파 
     optimizer.step() 
     # 역전파를 통해 얻은 gradient를 optimzer.step()을 통해 업데이트
     if batch_idx % 50 == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format( epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.data.item())) 

# test 과정
def test(): 
  model.eval() 
  test_loss = 0 
  correct = 0 
  for data, target in test_loader:
     data = data.to(device) 
     target = target.to(device) 
     output = model(data)
     test_loss += criterion(output, target).data.item()
     # get the index of the max log-probability 
     pred = output.data.max(1, keepdim=True)[1] 
     correct += pred.eq(target.data.view_as(pred)).cpu().sum() 
     
     test_loss /= len(test_loader.dataset)/batch_size 
     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format( test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

for epoch in range(1, 10):
# epoch는 총 train과 test를 몇 번 실행할지 결정
   train(epoch) 
   test()
# 9번의 train 이후의 test 값을 보게 되면 정확도가 98%에 이르는 걸 확인할 수 있다.
