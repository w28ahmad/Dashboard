from flask import Blueprint, abort, render_template, request, redirect, url_for, session
from flask_login import login_user, login_required, logout_user

corner_case_api = Blueprint('corner_case', __name__)

# Login redirect
@corner_case_api.route('/')
def goto_login():
    return redirect('/login')

# Route to logout
# Remove all the cookies
@corner_case_api.route('/logout')
@login_required
def signout():
    logout_user
    session.clear() #! BUG-FIX - One of the cookies still remains even after logout_user
    return redirect('/login')


