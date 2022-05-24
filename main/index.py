import datetime
import os
from pathlib import Path

from flask import Blueprint, jsonify, request

blue_index = Blueprint("index", __name__, url_prefix="/")


@blue_index.route('/upload', methods=['POST'])
def upload():
    print(request.files)
    file = request.files['file_give']             # 파일 받기
    extension = file.filename.split('.')[-1]      # 확장자
    today = datetime.datetime.now()                        # 현재 날짜, 시간
    mytime = today.strftime('%Y-%m-%d-%H-%M-%S')  #날짜 시간을 원하는 형태로 변환
    filename = f'{mytime}.{extension}'  # 파일 이름
    save_to = f'/static/img/{filename}' # 파일 경로
    # test = os.path.abspath(__file__)    # 절대경로

    # (1)프론트엔드 폴더로 절대경로 수정!!!!!
    abs_path = "C:/Users/USER/OneDrive/바탕 화면/p1_front" + save_to
    
    file.save(abs_path)                 # 파일 저장
    return jsonify({'result': 'success'})