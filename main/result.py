import io
import os
from matplotlib.pyplot import imshow
import numpy as np
import torch
from torchvision import datasets, models, transforms
from PIL import Image
from torch import nn

from flask import Blueprint, jsonify, request

blue_result = Blueprint("result", __name__, url_prefix="/")

# 모델안에 있던 변수들
device = "cpu"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet34(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)
checkpoint = torch.load('main/static/model/model.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load('main/static/model/model.pth', map_location=torch.device('cpu')))
# model = torch.load('main/static/model/model.pth', map_location=torch.device('cpu'))

model.eval()

# 이미지를 읽어 결과를 반환하는 함수
def get_prediction(image_bytes):
    image = Image.open(image_bytes).convert('RGB')
    transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transforms_test(image).unsqueeze(0).to(device)
    class_names = ['김수로', '이병헌', '송중기']

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        imshow(image.cpu().data[0], title='예측 결과: ' + class_names[preds[0]])

    return class_names[preds[0]]

@blue_result.route('/result', methods=['GET'])
def result():
    # 1. 폴더에 업로드된 파일을 불러와(최신파일?!)
    filenames = os.listdir('main/static/img/')
    file = filenames[-1]
    file_path = f'main/static/img/{file}'
    # 2. 
    output = get_prediction(file_path)
    # 3. 결과값을 DB에 저장
    doc = {
        # 유저아이디 토큰
        # 회원가입 시 유저 아이디
        # 파일 이름
        # 모델 결과값
    }
    return jsonify({'msg': 'success'})






    



    

    
