# -*- coding: utf-8 -*-
"""
Neural Networks(신경망)
===============

torch.nn 패키지를 사용하여 Neural Network를 만들 수 있다.

nn은 모델을 정의하고 미분하는 데 있어서 autograd에 의존한다.
nn.Module은 여러 계층으로 이뤄져있으며, 각 계층의 forward(input)메서드에 따라 output을 반환한다.

디지털 이미지를 분류하는 네트워크의 예시를 살펴보자:

<img src='https://pytorch.org/tutorials/_images/mnist.png'>
   convnet

위 그림은 단순한 feed-forward 네트워크로 이 네트워크는 input을 받고 그것을 몇개의 계층에 순차적으로 넘겨준다.
그리고 최종적으로 output을 출력한다.

일반적인 neural network 훈련절차는 다음과 같다:

- 학습가능한 몇개의 매개변수(또는 가중치)를 가진 neural network를 정의한다.
- 입력 데이터셋에 대한 반복학습을 진행한다.
- 네트워크를 통해 입력값을 처리한다.
- output이 정확한 값으로부터 얼마나 떨어져 있는지에 대한 손실값(loss)을 계산한다.
- 네트워크의 파라미터로 gradients를 역전파(backprop)한다.
- 단순한 업데이트 규칙에 따라 네트워크의 가중치를 업데이트한다.(weight = weigt - learning_rate * gradient)

Define the network(네트워크 정의)
------------------

그렇다면 훈련 절차에 따라 neural network를 정의해보자:
"""
import torch # 토치 라이브러리
import torch.nn as nn # nn관련 라이브러리
import torch.nn.functional as F # 활성화 함수(임계값을 기준으로 활성/비활성화 되는 함수) 등 다양한 함수를 포함


class Net(nn.Module): # Net가 nn.Module을 상속

    def __init__(self): # 초기화 단계 : neural network의 계층을 정의하는 단계
        super(Net, self).__init__() # 다량의 상속이 발생했을 경우를 방지
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        # Conv2d : Convolutional Network 2D
        self.conv1 = nn.Conv2d(1, 6, 3) # 1계층 : 속성의 상실을 방지하기 위해 1개의 채널을 3x3행렬의 6개 채널로 나눠줌. 
        self.conv2 = nn.Conv2d(6, 16, 3) # 2계층 : 6채널을 3x3의 16개 채널로 나눠줌
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        # 3계층 : Liear을 통한 직렬화(flat)된 배열로 채널이 없음.
        # 앞 계층의 출력을 flat해서 입력으로 받음.
        # 앞 계층에서 받은 16*6*6의 데이터를 120개로 압축해서 출력.
        self.fc2 = nn.Linear(120, 84) # 4계층 : 전달받은 120개의 데이터를 84개의 데이터로 바꿈.
        self.fc3 = nn.Linear(84, 10) # 5계층: 84개의 데이터를 최종적으로 10개로 출력

    def forward(self, x): # 계층을 이어주는 역할
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # relu() : 활성화 함수
        # Max_pool() : 최대값을 줄여주는 방법을 정해줌
        # conv1(x)를 받아와서 relu()에 돌려주고 이를 Max_pool()을 통해 사이즈를 줄여줌.
        # 화면을 (2, 2)픽셀로 받아서 Max_pooling을 통해 선택된 값으로 압축
        # 사이즈가 정사각형의 형태이므로 특정한 하나의 값만 있어도 된다.
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        # num_flat_features() : 여러 채널로 나눠진 데이터를 펴주는 역할
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x): # num_flat_features() : 여러 채널로 나눠진 데이터를 펴주는 역할
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

########################################################################
# gradients가 계산되는 backward 함수는 autograd를 사용하면 자동으로 정의되므로 forward 함수만 정의하면 된다.
# forward 함수에서는 Tensor의 모든 연산이 사용 가능하다.
#
# 모델의 학습가능한 파라미터는  net.parameters()를 통해 반환된다.

# 매개변수를 보여주는 시스템
params = list(net.parameters())
# list(net.parameters()) : 계층 간 통신에서 어떠한 알고리즘을 통해 연산을 했는지 보여줌.
print(len(params)) # params의 길이
print(params[0].size())  # conv1의 weight

########################################################################
# 랜덤한 32x32의 입력값을 넣어보자
# Note: expected input size of this net (LeNet) is 32x32. To use this net on
# MNIST dataset, please resize the images from the dataset to 32x32.

input = torch.randn(1, 1, 32, 32) # 32x32의 랜덤한 값을 입력값으로 설정.(dummy inputs)
out = net(input) # 네트워크에 학습
print(out) # 출력

########################################################################
# 모든 gradient buffer를 제로화하고, 랜덤한 gradients를 역전파(autograd의 동작):
net.zero_grad()
out.backward(torch.randn(1, 10))

########################################################################
# .. note::
# torch.nn은 mini-batch만을 지원합니다. torch.nn 패키지 전체는 mini-batch 형태인 입력만을 지원합니다. 단일 데이터는 입력으로 취급하지 않습니다. 예를 들어서, nn.Conv2d는 데이터건수 x 채널수 x 높이 x 너비 형식의 4D Tensor를 취합니다. 1개 데이터를 네트웍에 입력해야 한다면, input.unsqueexe(0)을 이용하여 가짜 임시 배치 차원을 추가해서 사용합니다.
#
#
# **Recap:**
#   -  torch.Tensor - backward()와 같은 autograd 연산을 지원하는 다차원 배열이며 텐서에 대한 gradient를 가지고 있다.
#   -  nn.Module - neural network 모듈로서 파라미터를 GPU로 옮기거나, 내보내기, 불러오기 등의 보조 작업을 이용하여 파라미터를 캡슐화하는 편리한 방법이다.
#   -  nn.Parameter - 텐서의 일종으로 Module에 속성으로 할당 될 때 파라미터로 자동으로 등록된다.
#   -  autograd.Function - autograd 연산의 forward와 backward에 대한 정의를 구현한다. 모든 텐서 연산은 최소한 하나의 Function 노드를 생성하는데, 이 노드는 텐서를 생성하고 기록을 인코딩하는 여러 함수들에 연결된다.
#
# **지금까지 한 것 :**
#   -  neural network 정의
#   -  입력값 처리와 backward 호출
#
# **아직 남아있는 것:**
#   -  loss(손실값) 계산
#   -  네트워크의 가중치(weight) 업데이트
#
# Loss Function(손실함수)
# -------------
# Loss Function은 (output, target)형태의 입력을 받아, output이 target에서 얼마나 멀리 떨어져 있는지를 추정하는 값을 계산한다.
#
# nn패키지의 몇몇 다른 `loss functions <https://pytorch.org/docs/nn.html#loss-functions>`과는 차이가 있으며
# 가장 간단한 Loss Function으로는 nn.MSELoss가 있으며 이는 입력과 정답 사이의 평균 제곱 오차(mean-squared error)를 계산한다.
#
# 예를 들면 다음과 같다 :

output = net(input) #network에 dummy inputs 넣어줌.
target = torch.randn(10)  # randn을 통해 생성한 10개의 배열을 dummy target으로 만들어줌.
target = target.view(1, -1)  # 결과 모양 맞추기.
criterion = nn.MSELoss() # Mean Squares Error : 데이터들의 오차가 서로 상쇄되는 것을 막기위해 오차에 대한 면적을 구해주는 것(제곱)

loss = criterion(output, target) #criterion에 대해 output과 target의 오차를 계산
print(loss)

########################################################################
# 만약 .grad_fn속성을 통해 역방향으로 loss를 추적한다면 다음과 같은 계산 그래프를 볼 수 있다:
#
#     input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
#           -> view -> linear -> relu -> linear -> relu -> linear
#           -> MSELoss
#           -> loss
#
# 따라서, loss.backward()를 호출하면 전체 그래프는 손실에 대하여 미분 계산이 수행되며, 그래프 내의 requires_grad = True인 모든 텐서들은 gradient로 누적된 .grad 텐서를 갖게 된다.
#
# 예를 들어 몇가지 단계를 역으로 진행하면 다음과 같다:

print(loss.grad_fn)  # MSELoss와 관련된 것은 grad_fn에 저장되어있음.
print(loss.grad_fn.next_functions[0][0])  # Linear과 관련된 것은 .next_functions[0][0]을 통해 알 수 있음.
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU와 관련된 것은 .next_functions[0][0].next_functions[0][0]을 통해 알 수 있음

########################################################################
# Backprop(역전파)
# --------
# 에러를 역전파하기 위해 사용자는 loss.backward()를 호출하면 된다. 다만 기존에 존재하는 gradients를 초기화할 필요가 있다. 그렇지 않으면 gradients가 기존의 gradients에 누적되어 저장되기 때문이다.
#
#
# loss.backward()를 호출하고 backward 호출 이전과 이후의 conv1's bias gradient를 살펴보자.


net.zero_grad() # 모든 파라미터의 gradient buffer를 제로화

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward() # loss.backward()를 호출

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

#########################################################################
# **마지막으로 남은 것:**
#
#   - Updating the weights of the network(네트워크의 가중치 업데이트)
#
# Update the weights(가중치 업데이트)
# ------------------
# 실제로 사용되는 가장 간단한 업데이트 규칙은 Stochastic Gradient Descent(SGD)이다:
#
#      weight = weight - learning_rate * gradient
#
# 간단한 파이썬 코드를 사용해 이를 구현할 수 있다: 
#
#     learning_rate = 0.01
#     for f in net.parameters():
#         f.data.sub_(f.grad.data * learning_rate)
#
# 그러나 neural network를 사용할 때 SGD, Nesterov-SGD, Adam, RMSProp 등과 같은 다양한 업데이트 규칙을 사용하고 싶을 수도 있다.
# 이러한 규칙의 사용을 위해 Pytorch는 이를 포함하는 torch.optim 패키지를 제공한다.
# 해당 패키지는 다음과 같이 매우 간단하게 사용할 수 있다:

import torch.optim as optim

# optimizer 생성
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # gradient buffers 제로
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # 업데이트

