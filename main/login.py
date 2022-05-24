import hashlib
import json
import jwt
from flask import Blueprint, jsonify, request
from pymongo import MongoClient
from datetime import datetime, timedelta

blue_login = Blueprint("login", __name__, url_prefix="/login")

client = MongoClient('localhost', 27017)
db = client.dbobject

SECRET_KEY = 'object'

@blue_login.route("/", methods=["POST"])
def log_in():
    data = json.loads(request.data)

    user_id = data.get("user_id")
    password = data.get("password")
    password_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()

    result = db.users.find_one({"user_id": user_id, "password": password_hash})

    if (user_id == ""):
        print("아이디를 입력해주세요.")
        return jsonify({'message' : 'id none'})
    elif (password == ""):
        print("비밀번호를 입력해주세요.")
        return jsonify({'message' : 'password none'})
    elif result is None:
        print("아이디 또는 비밀번호가 일치하지 않습니다.")
        return jsonify({'message' : 'id and password is different'})
    else:
        payload = {
            "id" : str(result["_id"]),
            "exp" : datetime.utcnow() + timedelta(seconds=60*60*24)
        }

        #token 체크용, 없어도 무방
        token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')

        print("로그인 완료")
        #return jsonify({'message' : 'success'})
        return jsonify({'message': 'success', 'token': token})