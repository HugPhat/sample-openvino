import os

import argparse

from flask import Flask, Response, render_template, request, jsonify, url_for, redirect
from werkzeug.utils import secure_filename
import cv2

from main import app_run
import config 


app = Flask(__name__)

MODE = os.environ.get('FLASK_ENV', 'development')
app = Flask(__name__)
if MODE == 'production':
    app.config.from_object(config.ProductionConfig)
elif MODE == 'development':
    app.config.from_object(config.DevelopmentConfig)

CAMERA = None
RESULT = None
ThumbImage = cv2.imread('public/thumb.png')

def gen():
    global CAMERA
    global RESULT
    while True:
        if not CAMERA is None:
            flag, frame = CAMERA.read()
            if flag:
                RESULT = model.run(frame,draw=True)
                ret, jpeg = cv2.imencode(".jpg", frame)
                frame = jpeg.tostring()
                yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")
        else:
            CAMERA = None
            ret, jpeg = cv2.imencode(".jpg", ThumbImage)
            frame = jpeg.tostring()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")

@app.route("/")
def index():
    return render_template(
        "index.html", is_async=True, enumerate=enumerate,
    )

@app.route("/video_feed")
def video_feed():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/result", methods=["GET"])
def result():
    if not RESULT is None:
        return jsonify(result= RESULT)
    else:
        return jsonify(result='')

#################
app.config["VIDEO_UPLOADS"] = "cache"
app.config["ALLOWED_VIDEO_EXTENSIONS"] = ["MP4"]
app.config["MAX_FILESIZE"] = 10* 1024 * 1024
def allowed_type(filename):
    if not "." in filename:
        return False
    ext = filename.rsplit(".", 1)[1]
    if ext.upper() in app.config["ALLOWED_VIDEO_EXTENSIONS"]:
        return True
    else:
        return False
def allowed_filesize(filesize):
    if int(filesize) <= app.config["MAX_FILESIZE"]:
        return True
    else:
        return False

@app.route("/upload", methods=["POST"])
def upload_image():
    global CAMERA
    if request.files:
            video = request.files["video"]
            if video.filename == "":
                app.logger.info("No filename")
                return redirect(request.url_root)
            if allowed_type(video.filename):
                if not CAMERA is None:
                    CAMERA.release()
                    cv2.destroyAllWindows()
                filename = os.path.join(app.config["VIDEO_UPLOADS"], 'video.mp4')#secure_filename(video.filename)
                video.save(filename)#os.path.join(app.config["VIDEO_UPLOADS"], filename))
                model.reset()
                CAMERA = cv2.VideoCapture(filename)
                return redirect(request.url_root)
            else:
                app.logger.info("That file extension is not allowed")
                return redirect(request.url_root)
    return redirect(request.url_root)

if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--age_gender', required=False,action='store_true', help='run age gender model only')
    parser.add_argument('--person_reid', required=False,action='store_true', help='run person reid only')
    args = parser.parse_args()

    if not args.age_gender and not args.person_reid:
        model = app_run.both()  
    elif args.age_gender:
        model = app_run.face_age_gender()
    else:
        model = app_run.person_reid()

    #camera = cv2.VideoCapture('videos/t4.mp4')

    from waitress import serve

    serve(app, host=app.config['HOST'], port=app.config['PORT'])