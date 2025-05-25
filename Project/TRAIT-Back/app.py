from flask import Flask
from flask_cors import CORS

from routes.image_routes import image_blueprint
from routes.video_routes import video_blueprint

def create_app():
    app = Flask(__name__)
    CORS(app)

    app.register_blueprint(image_blueprint)
    app.register_blueprint(video_blueprint)

    return app