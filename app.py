from pathlib import Path
import datetime
from flask import Flask, jsonify, request
import io
from PIL import Image
import matplotlib as mat
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import time
import os
import shutil
from bing_image_downloader import downloader  # 크롤링용

mat.rcParams['font.family'] = 'New Gulim'

directory_list = [
    './custom_dataset/train/',
    './custom_dataset/test/',
]

# 초기 디렉토리 만들기
for directory in directory_list:
    if not os.path.isdir(directory):
        os.makedirs(directory)


# 수집한 이미지를 학습 데이터와 평가 데이터로 구분하는 함수
def dataset_split(query, train_cnt):
    # 학습 및 평가 데이터셋 디렉토리 만들기
    for directory in directory_list:
        if not os.path.isdir(directory + '/' + query):
            os.makedirs(directory + '/' + query)
    # 학습 및 평가 데이터셋 준비하기
    cnt = 0
    for file_name in os.listdir(query):
        if cnt < train_cnt:
            print(f'[Train Dataset] {file_name}')
            shutil.move(query + '/' + file_name,
                        './custom_dataset/train/' + query + '/' + file_name)
        else:
            print(f'[Test Dataset] {file_name}')
            shutil.move(query + '/' + file_name,
                        './custom_dataset/test/' + query + '/' + file_name)
        cnt += 1
    shutil.rmtree(query)

###########################################################################################
## 이미지 파일 크롤링 해두는 부분입니다. 주석 풀어서 쓰시면되고 한번 크롤링 후에는 다시 주석처리 ~~
###########################################################################################

query = '너구리'
downloader.download(query, limit=10, output_dir='./', adult_filter_off=True, force_replace=False, timeout=60)
dataset_split(query, 10)

query = '김수로'
downloader.download(query, limit=10, output_dir='./', adult_filter_off=True, force_replace=False, timeout=60)
dataset_split(query, 10)

query = '고양이'
downloader.download(query, limit=40, output_dir='./', adult_filter_off=True, force_replace=False, timeout=60)
dataset_split(query, 10)

device = torch.device("cuda:0" if torch.cuda.is_available()else "cpu")  # device 객체
print(device)

###################################################################

transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # 데이터 증강
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [
                         0.229, 0.224, 0.225])  # 정규화(normalization)
])

transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

###################################################################

data_dir = './custom_dataset'
train_datasets = datasets.ImageFolder(
    os.path.join(data_dir, 'train'), transforms_train)
test_datasets = datasets.ImageFolder(
    os.path.join(data_dir, 'test'), transforms_test)

train_dataloader = torch.utils.data.DataLoader(
    train_datasets, batch_size=4, shuffle=True, num_workers=0)
test_dataloader = torch.utils.data.DataLoader(
    test_datasets, batch_size=4, shuffle=True, num_workers=0)

print('학습 데이터셋 크기:', len(train_datasets))
print('테스트 데이터셋 크기:', len(test_datasets))

class_names = train_datasets.classes
print('클래스:', class_names)

###################################################################

def imshow(input, title):
    # torch.Tensor를 numpy 객체로 변환
    input = input.numpy().transpose((1, 2, 0))
    # 이미지 정규화 해제하기
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    # 이미지 출력
    plt.imshow(input)
    plt.title(title)
    plt.show()


# 학습 데이터를 배치 단위로 불러오기
iterator = iter(train_dataloader)

# 현재 배치를 이용해 격자 형태의 이미지를 만들어 시각화
inputs, classes = next(iterator)
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])

###################################################################

model = models.resnet34(pretrained=True)

num_features = model.fc.in_features

# 전이 학습(transfer learning): 모델의 출력 뉴런 수를 3개로 교체하여 마지막 레이어 다시 학습
model.fc = nn.Linear(num_features, 3)  # 쿼리 갯수
model = model.to(device) # 모델을 gpu로 연산

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 1
model.train()
start_time = time.time()

# 전체 반복(epoch) 수 만큼 반복하며
for epoch in range(num_epochs):
    running_loss = 0.
    running_corrects = 0

    # 배치 단위로 학습 데이터 불러오기
    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 모델에 입력(forward)하고 결과 계산
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # 역전파를 통해 기울기(gradient) 계산 및 학습 진행
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_datasets)
    epoch_acc = running_corrects / len(train_datasets) * 100.

    # 학습 과정 중에 결과 출력
    print('#{} Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(
        epoch, epoch_loss, epoch_acc, time.time() - start_time))


torch.save(model.state_dict(), 'static/model/model.pth')

model.eval()
start_time = time.time()

with torch.no_grad():
    running_loss = 0.
    running_corrects = 0

    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        # 한 배치의 첫 번째 이미지에 대하여 결과 시각화
        print(
            f'[예측 결과: {class_names[preds[0]]}] (실제 정답: {class_names[labels.data[0]]})')
        imshow(inputs.cpu().data[0], title='예측 결과: ' + class_names[preds[0]])

    epoch_loss = running_loss / len(test_datasets)
    epoch_acc = running_corrects / len(test_datasets) * 100.
    print(
        '[Test Phase] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch_loss, epoch_acc, time.time() - start_time))

# app = Flask(__name__)

# image = Image.open('static/ehdtjr.jpg')
# image = transforms_test(image).unsqueeze(0).to(device)

# with torch.no_grad():
#     outputs = model(image)
#     _, preds = torch.max(outputs, 1)
#     imshow(image.cpu().data[0], title='예측 결과: ' + class_names[preds[0]])

# # 이미지를 읽어 결과를 반환하는 함수
# def get_prediction(image_bytes):
#     image = Image.open(io.BytesIO(image_bytes))
#     image = transforms_test(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         outputs = model(image)
#         _, preds = torch.max(outputs, 1)
#         imshow(image.cpu().data[0], title='예측 결과: ' + class_names[preds[0]])

#     return class_names[preds[0]]


# @app.route('/', methods=['POST'])
# def predict():
#     if request.method == "POST":
#         file = request.files['file']
#         extension = file.filename.split('.')[-1]
#         today = datetime.datetime.now()
#         mytime = today.strftime('%Y-%m-%d-%H-%M-%S')
#         filename = f'{mytime}.{extension}'
#         save_to = f'/static/img/{filename}'
#         test = os.path.abspath(__file__)
#         # print(test)
#         parent_path = Path(test).parent
#         # print(parent_path)
#         abs_path = str(parent_path) + save_to
#         # print("abs_path", end=""), print(abs_path)
#         file.save(abs_path)

#         image = Image.open(abs_path)
#         image = transforms_test(image).unsqueeze(0).to(device)

#         with torch.no_grad():
#             outputs = model(image)
#             _, preds = torch.max(outputs, 1)
#             print(class_names[preds[0]]) ## 예측 결과
#             print(
#                 f'[예측 결과: {class_names[preds[0]]}] (실제 정답: {class_names[labels.data[0]]})')
#             imshow(inputs.cpu().data[0], title='예측 결과: ' + class_names[preds[0]])

#         return jsonify({'class_name': "zd"})


# app.run(port=5001)