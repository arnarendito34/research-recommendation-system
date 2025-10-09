from flask import Flask
app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return 'index page'

@app.route('/hi')
def hi():
    return 'olaa'