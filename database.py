from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

# Model of users
class User(UserMixin, db.Model):
    __tablename__ = 'Clients'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100))
    password = db.Column(db.String(100))
    
# db.create_all()
# db.session.commit()