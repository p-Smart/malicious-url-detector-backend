from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def hello_world():
    response = {
        "success": True,
        "message": "Hello world."
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)