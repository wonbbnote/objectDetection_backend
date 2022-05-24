import hashlib
import json
from flask import Blueprint, jsonify, request
from pymongo import MongoClient

blue_signup = Blueprint("signup", __name__, url_prefix="/signup")

client = MongoClient('localhost', 27017)
db = client.dbobject

@blue_signup.route("/", methods=["POST"])
def sign_up():
    data = json.loads(request.data)

    password = data['password']
    password_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()

    user_id = data['user_id']
    password_check = data['password_check']
    user_info = bool(db.users.find_one({"user_id" : data['user_id']}))

    if (user_id == ""):
        print("아이디를 입력해주세요.")
        return jsonify({'message' : 'id none'})
    elif (password == ""):
        print("비밀번호를 입력해주세요.")
        return jsonify({'message' : 'password none'})
    elif (password_check == ""):
        print("비밀번호 확인이 필요합니다.")
        return jsonify({'message' : 'password check none'})
    elif (password != password_check):
        print("비밀번호가 일치하지 않습니다.")
        return jsonify({'message' : 'password is different'})
    else:
        if not user_info:
            doc = {
                    'user_id' : data['user_id'],
                    'password' : password_hash
                }
            db.users.insert_one(doc)

            print("회원가입 완료")
            return jsonify({'message' : 'success'})
        else:  
            return jsonify({'message' : 'id is duplicated'})