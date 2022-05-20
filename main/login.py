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

    if result is None:
        return jsonify({'message' : '아이디 혹은 비밀번호가 옳지 않습니다.'}), 401

    payload = {
        "id" : str(result["_id"]),
        "exp" : datetime.utcnow() + timedelta(seconds=60*60*24)
    }

    token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')

    return jsonify({'message': 'login', 'token': token})