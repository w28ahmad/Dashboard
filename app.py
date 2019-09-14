from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', name="Wahab")

@app.route('/login/<username>')
def login(username):
    return f"The username of the user is: {username}"

@app.route('/post/<int:postId>')
def post(postId):
    return f"Your post Id is: {postId}"


if __name__ == "__main__":
    app.run(debug=True)
         
