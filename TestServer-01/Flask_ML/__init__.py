from flask import Flask

def create_app():
    app = Flask(__name__)
    
    from Flask_ML.routes import (main_route, predict_route)
    app.register_blueprint(main_route.bp)
    app.register_blueprint(predict_route.bp, url_prefix='/predict')

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
