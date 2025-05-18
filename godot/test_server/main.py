from flask import Flask, jsonify, request
import random

app = Flask(__name__)

@app.route('/', methods=['GET'])
def allive():
    return "allive"

@app.route('/reasoning', methods=['POST'])
def reasoning():
    print(request.json)
    print(request.json["view_item"][49])
    exec_type = input("1: move; 2: pick > ")
    if exec_type == "1":
        a = input("x> ")
        b = input("y> ")
        data = {
            "id" : 1,
            "next": {
                "kind": "move",
                "x": int(a),
                "y": int(b)
            }
        }
        return data
    elif exec_type == "2":
        item = input("itemid> ")
        data = {
            "id" : 1,
            "next": {
                "kind": "pick",
                "item": int(item)
            }
        }
        return data

if __name__ == '__main__':
    app.run(host='localhost', port=8000)