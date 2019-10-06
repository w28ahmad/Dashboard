from flask_sqlalchemy import SQLAlchemy

class User(db.Model):
    __tablename__ = 'Clients'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80))
    password = db.Column(db.String(80))
    
# db.create_all()
# db.session.commit()