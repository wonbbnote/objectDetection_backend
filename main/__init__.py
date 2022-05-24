# '__init__.py'는 폴더의 파이썬 파일 중 가장 먼저 실행 
# app이라는 flask 객체를 만들어 여기서 blueprint 등록해주는 작업 수행(register_blueprint)
from flask import Flask
from flask_cors import CORS
from flask import Blueprint, jsonify, request
import torch
#수정
from . import signup
from . import login
#수정
from . import index
from . import result

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})

#수정
app.register_blueprint(signup.blue_signup)
app.register_blueprint(login.blue_login)
#수정
app.register_blueprint(index.blue_index)
app.register_blueprint(result.blue_result)
