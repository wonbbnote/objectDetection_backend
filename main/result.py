from functools import wraps
import io
import json
import os
from bson import ObjectId
from matplotlib.pyplot import imshow
import numpy as np
import torch
from torchvision import datasets, models, transforms
from PIL import Image
from torch import nn
import jwt
from flask import Blueprint, abort, jsonify, request
from pymongo import MongoClient


client = MongoClient('localhost', 27017)
db = client.dbobject

SECRET_KEY = 'object'

blue_result = Blueprint("result", __name__, url_prefix="/")

# 모델안에 있던 변수들
device = "cpu"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet34(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
checkpoint = torch.load('main/static/model/sgd10-4.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load('main/static/model/model.pth', map_location=torch.device('cpu')))
# model = torch.load('main/static/model/model.pth', map_location=torch.device('cpu'))

model.eval()

# 데코레이터 함수
def authorize(f):
    @wraps(f)
    def decorated_function():
        if not 'Authorization' in request.headers:
            abort(401)
        token = request.headers['Authorization']
        try:
            user = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        except:
            abort(401)
        return f(user)
    return decorated_function

# 이미지를 읽어 결과를 반환하는 함수
def get_prediction(image_bytes):
    image = Image.open(image_bytes).convert('RGB')
    print(image_bytes)
    print(image)
    transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#x
    ])
    image = transforms_test(image).unsqueeze(0).to(device)
    class_names = ['신동엽', '정상훈', '안영미', '김민교', '권혁수', '정이랑', '정혁', '주현영', '이소진', '솔빈']

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        #imshow(image.cpu().data[0], title='예측 결과: ' + class_names[preds[0]])

    return class_names[preds[0]]

@blue_result.route('/result', methods=['GET'])
def get_result():
    # 1. 폴더에 업로드된 파일을 불러와(최신파일?!)
    # (2)프론트엔드 폴더로 절대경로 수정!!!!!
    filenames = os.listdir('C:/Users/USER/OneDrive/바탕 화면/p1_front/static/img/')
    print(filenames)
    file = filenames[-1]
    # (3)프론트엔드 폴더로 절대경로 수정!!!!!
    abs_path = f'C:/Users/USER/OneDrive/바탕 화면/p1_front/static/img/{file}'
    file_path = f'/static/img/{file}'
    
    # 2. 모델에 적용
    # image = 'main/static/img/2022-05-23-09-46-11.png'
    output = get_prediction(abs_path)
    print(output)
    return jsonify({'msg': 'success', 'output': output, 'file_path': file_path})
    

@blue_result.route('/result', methods=['POST'])
@authorize
def post_result(user):
    filenames = os.listdir('main/static/img/')
    file = filenames[-1]
    file_path = f'main/static/img/{file}'
    output = get_prediction(file_path)
    # token = request.headers.get("Authorization")
    # user = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
    db_user = db.users.find_one({'_id': ObjectId(user.get("id"))}) #?
    # 3. 결과값을 DB에 저장
    doc = {
        # 회원가입 시 유저 아이디
        "user_id": db_user['user_id'],
        # 파일 이름
        "filename": file_path,
        # 모델 결과값
        "output": output
    }
    db.result.insert_one(doc)
    return jsonify({'message' : 'success'})
    
