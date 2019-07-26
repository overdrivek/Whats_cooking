from app import app
from flask import render_template

@app.route('/')
@app.route('/index')
def index():
    user = {'username':'Krishna'}
    posts = [
        {
            'author': {'username': 'John'},
            'body': 'Wynonas Big Brown Beaver!'
        },
        {
            'author': {'username': 'Susan'},
            'body': 'Southbound pachyderm'
        }
    ]
    return render_template('index.html',title='Home',user=user,posts=posts)
