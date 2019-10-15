신경망을 사용하는 기계학습 중 CNN은 이미지 프로세싱 등의 분야에서 좋은 성능을 발휘한다.
CNN은 기본적으로 nxn 모양의 필터 모양을 가지고 이미지의 부분을 자른 후 특징을 뽑아낸다는 특징이 있다.

Pytorch에서 제공하는 conv2d 함수는 data가 몇 겹의 데이터로 구성되어있는지를 나타내는 in_channel(cifar10과 같은 컬러 사진의 경우 RGB를 사용하기 때문에 in_channel이 3, MNIST와 같은 흑백 사진의 경우 1), 출력할 conv layer의 층을 결정하는 out_channel, 마지막으로 필터의 사이즈를 결정하는 kernal size를 입력받아 실행된다. 

![대체 텍스트](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile3.uf.tistory.com%2Fimage%2F99EC1C435B48B8481B98EC)

위의 사진을 보면 가로와 세로가 각각 32이고, 깊이가 3인 이미지가 있고, 크기가 5x5인 3겹 필터를 사용하여 activation map을 만들고 있다.
이렇게 만들어진 activation map은 28x28개의 픽셀로 이뤄져있는데 각 픽셀의 갯수는 필터의 사이즈인 5x5x3= 75이다.

이렇게 하나의 이미지에 대해 activation map을 만든 후 Pooling이라는 연산을 사용하여 activation map의 의미있는 특징들을 추출한다.

![대체 텍스트](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile29.uf.tistory.com%2Fimage%2F991526445B49831101DC96)

위의 사진은 Max Pooling이라는 기법으로 activation map을 일정한 사이즈로 나눈 후 나눠진 영역에서 가장 큰 값을 대표값으로 설정하여 정해진 사이즈의 필터에 특징을 추출해낸다.
Pooling의 방법으로는 해당 구역의 평균을 뽑아내는 average pooling과 최솟값을 뽑아내는 Min pooling등의 방법도 존재한다.

![대체 텍스트](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile21.uf.tistory.com%2Fimage%2F999AB9425B498822317983)

이런 방식으로 Conv layer를 생성하고 그 conv layer를 pooling layer를 만든 뒤, 다시 pooling layer에 대해 conv layer를 생성하고 그 결과를 pooling하는 것을 반복하여 마지막에 fully connect layer에서 특징들을 유의미하게 분류하는 과정을 거친다. 

![대체 텍스트](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile29.uf.tistory.com%2Fimage%2F99E0713D5B498D4A1FF2D4)

이 과정을 Pytorch를 이용하여 구현하면 위의 사진과 같다. 

__init__ 함수의 fully connect layer에서는 neural network의 Linear함수를 통해 추출한 픽셀들을 0~9까지의 총 10개의 라벨에 따라 분류한다. 추출한 픽셀의 갯수는 수학적으로 계산하기보다는 아무 값이나 대입한 후 런타임 에러에서 말해주는 값으로 입력하면 된다. 

![대체 텍스트](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile30.uf.tistory.com%2Fimage%2F992ABA455B45E74E18AD88)

그리고 forward 함수에서는 마지막에 fully requested layer까지 거친 특징 값들을 위와 같은 softmax 함수를 통해 사람이 인식하기 쉬운 분류 값으로 출력받는다.

위의 과정을 train 과정을 거쳐 test해보면 MNIST에 대한 인식의 정확도가 96% 정도에 이름을 알 수 있다.

이러한 CNN의 성능을 더욱 높일 수 있도록 고안한 것이 Advanced CNN이다.
![대체 텍스트](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile27.uf.tistory.com%2Fimage%2F999A7A3F5B49B62125233E)

Advanced CNN에서는 하나의 이미지를 다양한 필터들로 conv layer를 만든 후에 그 layer들을 하나로 합친 다음 pooling layer를 만들어서 더욱 유의미한 특징 값을 추출하는 방식이다. 하나의 pooling layer를 만들 때 5x5, 3x3, 1x1 등의 다양한 커널 사이즈의 필터를 사용하기 때문에 conv layer를 만드는 과정에서 padding과 stride를 적절히 설정해줘야한다.


Advanced CNN의 핵심은 바로 conv layer를 만드는 과정에서 1x1 필터를 사용한다는 점이다. 
![대체 텍스트](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile5.uf.tistory.com%2Fimage%2F9933A3475B49C4F90A1CE7)

사진과 같이 커널 사이즈가 1인 필터를 사용하고 out_channel을 원래 이미지 in_chnnel의 절반인 32로 줄인 다음, 이 conv layer를 기반으로 다른 conv layer를 만들게 되면, 원래 이미지와 같은 width와 height의 이미지임에도 추후 conv layer를 만드는 연산의 양이 효과적으로 줄어들기 때문이다.

예를 들어, 28x28의 192체널로부터 5x5 conv layer 32체널을 만든다면 5x5x192x32의 파라미터가 만들어지겠지만,  1x1로 32체널로 줄인다면 줄이는 데에는 1x1x192x32의 파라미터가, conv layer엔 5x5x32x32의 파라미터가 필요하다.

즉, 기존에는 (5x5x28x28x192x32) 개의 연산이 필요 한 상황이라면 이제는 (1x1x28x28x192x32) + (28x28x5x5x32x32) 개의 연산이 필요하다.

![대체 텍스트](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile30.uf.tistory.com%2Fimage%2F997412365B49D6DD25877C)

이렇게 여러 개의 conv layer를 합쳐서 하나의 pooling layer로 바꾸는 과정을 Inception Module이라고 하는데, 이러한 inception module들을 이어붙이면 기존의 CNN보다 더 의미있는 특징 추출과 라벨 분류 학습이 가능하다는 게 Advanced CNN의 아이디어이다. 

Pytorch를 이용한 Advanced CNN의 구현을 아래 코드에서 설명해보겠다.
