import os
from flask import Flask, render_template, request
from pymongo import MongoClient
from flask_pymongo import PyMongo
import final
app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['MONGO_DBNAME'] = "axis_db"
app.config['MONGO_URI'] = "mongodb://localhost:27017/axis_db"
mongo = PyMongo(app)

@app.route("/")
def helloWorld():
    return render_template("index.html")

@app.route("/handle_data", methods=['POST'])
def checking():
    username = request.form['username']
    password = request.form['password']
    # post = {'username': username, 'password': password}
    user = mongo.db.users
    check = user.find_one({'username': username, 'password' : password})
    if check is None:
        return "username or password is incorrect :("
    return render_template("upload.html")

@app.route("/signUpform", methods=['POST'])
def signUPChecking():
    name = request.form['name']
    username = request.form['username']
    password = request.form['password']
    post = {'name': name, 'username': username, 'password': password}
    user = mongo.db.users
    find = user.find_one({'username': username})
    if find is not None:
        return "sorry username already exists :("
    user.insert(post)
    return render_template("index.html")

@app.route("/upload")
def showingUpload():
    return render_template("upload.html")

@app.route("/uploaded", methods=['POST'])
def uploadHandling():
    target = os.path.join(APP_ROOT, 'userimages/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)
    for file in request.files.getlist("file"):
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)

    if os.stat(destination).st_size > 1000011:
        os.remove(destination)
        return "Sorry the size of the image is greater than 1 mb :("
    results = final.probability(destination)
    genuine = int(results[0][1]*100)
    forged = int(results[0][0]*100)
    return render_template("image.html", genuine=genuine, forged=forged)



@app.route("/signup")
def signupPage():
    return render_template("signup.html")

if __name__ == "__main__":
    app.run(debug=True)

