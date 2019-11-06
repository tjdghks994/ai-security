#Torchvision Object Detection Finetunning Tutorial(Torchvision 객체감지 미세조정 튜토리얼)

#보행자 감지 및 분할을 위해 Penn-Fudan 데이터베이스에서 훈련된 마스크 R-CNN 모델을 미세 조정

#데이터셋 정의하기
import os               # 파일이나 디렉토리 경로에 관한 함수를 가진 모듈. 이미지 경로를 찾아가기 위해 쓰임.
import numpy as np      # numpy라이브러리를 import하고 이후부터는 
import torch            # torch 라이브러리
from PIL import Image   # PIL은 이미지 처리 패키지를 의미.
                        # PIL에서 제공하는 Image클래스는 다양한 포맷의 이미지를 읽고 변환하여 저장할 수 있도록 기능을 제공.


class PennFudanDataset(object): # PennFudanDataset이라는 클래스 선언.
    def __init__(self, root, transforms): # 생성자 정의
        self.root = root
        self.transforms = transforms
        # 모든 이미지파일 로드 후 정렬.
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages")))) # 'PNGImages'폴더의 모든 이미지를 로드하고 정렬.
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks")))) # 'PedMasks'폴더의 모든 이미지를 로드하고 정렬.

"""
데이터셋을 구성하기 위한 이미지데이터들의 구조
PennFudanPed/
  PedMasks/
    FudanPed00001_mask.png
    FudanPed00002_mask.png
    FudanPed00003_mask.png
    FudanPed00004_mask.png
    ...
  PNGImages/
    FudanPed00001.png
    FudanPed00002.png
    FudanPed00003.png
    FudanPed00004.png
"""

    def __getitem__(self, idx): # '__getiten__'은 매직매서드로서 객체에서 []연산자를 사용할 때의 동작을 정의.
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx]) # 각 경로를 병합하여 새로운 경로를 생성하고 변수 img_path에 넣음.
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx]) # 각 경로를 병합하여 새로운 경로를 생성하고 변수 mask_path에 넣음.
        img = Image.open(img_path).convert("RGB") # image.open()을 통해 사진파일을 열고, convert('RGB')를 통해 이미지를 삼원색을 이용해서 표현.
        mask = Image.open(mask_path) # mask의 경우 각 색상은 0이 배경인 다른 인스턴스에 해당하므로 RGB로 변경하지 않았음.
        mask = np.array(mask) # PIL모듈로 연 image를 numpy array로 바꿔줌.
        obj_ids = np.unique(mask) # 인스턴스들을 서로다른 색상으로 인코딩함.
        obj_ids = obj_ids[1:] # 첫번째 id는 배경이므로 제거.

        masks = mask == obj_ids[:, None, None] # 각 색상으로 인코딩한 mask를 일련의 이진 mask로 쪼갬.

        # 각 mask에 대한 bounding box 좌표 가져오기
        num_objs = len(obj_ids) # obj_ids의 길이를 num_objs에 넣어줌.
        boxes = [] # boxes를 배열로 선언
        for i in range(num_objs):       # num_objs의 범위 내에서 i가 1씩 증가할 때
            pos = np.where(masks[i])    # masks[i]의 index를 찾고 pos에 넣어줌.
            xmin = np.min(pos[1])       # pos[1]의 최솟값을 xmin에 넣어줌.
            xmax = np.max(pos[1])       # pos[1]의 최댓값을 xmax에 넣어줌.
            ymin = np.min(pos[0])       # pos[0]의 최솟값을 ymin에 넣어줌.
            ymax = np.max(pos[0])       # pos[0]의 최댓값을 ymax에 넣어줌. 
            boxes.append([xmin, ymin, xmax, ymax]) # boxes[i]에 [xmin, ymin, xmax, ymax]를 넣어줌.

        # torch.Tensor로 변환하기
        boxes = torch.as_tensor(boxes, dtype=torch.float32) # torch.as_tensor를 통해 ndarray객체인 boxes를 받고, 값 참조(refernce)를 사용하여 텐서 자료형 뷰(view)를 만듦.
        
        labels = torch.ones((num_objs,), dtype=torch.int64) # (num_objs, ) 사이즈의 모든 요소가 1로 이루어진 텐서 labels생성.
        masks = torch.as_tensor(masks, dtype=torch.uint8) # torch.as_tensor를 통해 ndarray객체인 masks를 받고, 값 참조(refernce)를 사용하여 텐서 자료형 뷰(view)를 만듦.

        image_id = torch.tensor([idx]) # 값 복사(value copy)를 사용하여 [idx]값을 복사한 새로운 텐서 자료형 인스턴스 image_id를 생성.
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # (ymax-ymin)와 (xmax-xmin)을 곱한 사각형의 면적을 area에 넣어줌.
        
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64) # (num_objs, )사이즈의 모든 요소가 0으로 이루어진 텐서 iscrowd를 생성하여 모든 인스턴스가 군중상태가 아니라고 가정.

        target = {}                     # target을 딕셔너리로 생성하고 각 필드에 값을 할당.
        target["boxes"] = boxes         # target["boxes"]에 boxes 값 할당.
        target["labels"] = labels       # target["lables"]에 lables 값 할당.
        target["masks"] = masks         # target["masks"]에 masks 값 할당.
        target["image_id"] = image_id   # target["image_id"]에 image_id 값 할당.
        target["area"] = area           # target["area"]에 area 값 할당.
        target["iscrowd"] = iscrowd     # target["iscrowd"]에 iscrowd 값 할당.

        if self.transforms is not None: 
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self): # '__len__'은 매직매서드로서 객체의 길이를 0이상의 정수로 반환. len()으로 호출.
        return len(self.imgs)

# Torchvision 모델주(model zoo, 역자주:미리 학습된 모델들을 모아 놓은 공간)에서 사용 가능한 모델들 중 하나를 이용해 모델을 수정하는 방법 중
# 미리 학습된 모델에서 시작해서 마지막 레이어 수준만 미세 조정하는 방법 사용.

# PennFudan 데이터셋을 위한 인스턴스 분할 모델
import torchvision                                                      # torchvision 모듈 사용.
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor  # torchvision에서 detection을 위한 모델인 faster_rcnn 모델에서 FastRCNNPredictor 클래스 사용.
                                                                        # Faster R-CNN은 이미지에서 객체에 대한 바운딩 박스와 객체에 대한 각 클래스별 신뢰도 점수까지 모두 예측하는 모델
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor    # torchvision에서 detection을 위한 모델인 mask_rcnn 모델에서 MaskRCNNPredictor 클래스 사용.
                                                                        # Mask R-CNN은 Faster R-CNN에 각 인스턴스에 대한 분할 마스크 예측하는 추가 분기(레이어)를 추가한 모델.
                                                                        # 즉 한 두 사물이 겹쳐 있더라도 서로 다른 사물로 분할하여 인식하는 모델


def get_model_instance_segmentation(num_classes): # get_model_instance_segmentation이라는 도움 함수 생성.
    # COCO 에서 미리 학습된 인스턴스 분할 모델을 로드
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # 분류기를 위한 입력 특징 차원을 획득.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # 미리 학습된 헤더를 새로운 것으로 바꿈.
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 마스크 분류기를 위한 입력 특징들의 차원을 획득.
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    
    # 마스크 예측기를 새로운 것으로 바꿈.
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

# 하나로 합치기
# 데이터 증강 / 변환을 위한 도움 함수 작성
import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor()) #  numpy 배열의 이미지를 torch 텐서로 바꾸어줌.
    if train:
        # 학습시 50% 확률로 학습 영상을 좌우 반전 변환.
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)    # 데이터들의 Normalization을 위해 .Compose()함수 사용.
                                    # .Compose()는 여러 transfoms들을 chaining.

# 학습과 검증 수행을 위한 메인함수 작성
from engine import train_one_epoch, evaluate
import utils


def main():
    # 학습을 GPU로 진행하되 GPU가 가용하지 않으면 CPU로 학습을 진행.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2 # 데이터셋이 배경과 사람 두가지 클래스만 가지므로 num_classes 값에 2를 할당.
    
    # 데이터셋과 정의된 transformation들을 사용하도록 함.
    dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
    dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))

    indices = torch.randperm(len(dataset)).tolist()                     # 데이터셋을 학습용과 테스트용으로 나눔.
    dataset = torch.utils.data.Subset(dataset, indices[:-50])           # 학습에 사용할 데이터(50개).
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:]) # 테스트에 사용할 데이터(50개).

    # 데이터 로더를 학습용과 검증용으로 정의.
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn) # 학습용 데이터

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn) # 검증용 데이터

    # 만들어둔 get_model_instance_segmentation(num_classes) 도움 함수를 이용해 모델을 가져옴.
    model = get_model_instance_segmentation(num_classes)

    # 모델을 GPU나 CPU로 옮김.
    model.to(device)

    # 옵티마이저(Optimizer).
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    
    # 학습률 스케쥴러.
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # 10 에포크만큼 학습.
    num_epochs = 10

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10) # 1 에포크동안 학습하고, 10회 마다 출력.
        lr_scheduler.step() # 학습률 업데이트.
        evaluate(model, data_loader_test, device=device) # 테스트 데이터셋에서 평가.

    print("That's it!")








