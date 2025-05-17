from flask import Flask, jsonify
import random

app = Flask(__name__)

@app.route('/', methods=['GET'])
def root_ret():
    data = {
        "id" : 1,
        "next": {
            "kind": "move",
            "x": random.randint(1, 10),
            "y": random.randint(1, 10)
        }
    }
    return data

if __name__ == '__main__':
    app.run(host='localhost', port=8000)