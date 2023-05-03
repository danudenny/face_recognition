import base64
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from flask import Flask, jsonify, make_response, request
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

# Load Face Detection Model
face_cascade = cv2.CascadeClassifier(
    "./models/haarcascade_frontalface_default.xml")
# Load Anti-Spoofing Model graph
json_file = open('antispoofing_models/antispoofing_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load antispoofing model weights
model.load_weights('antispoofing_models/antispoofing_model.h5')
print("Model loaded from disk")


def detect_liveness(frame):
    frame = cv2.resize(frame, (600, 400))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = frame[y-5:y+h+5, x-5:x+w+5]
        resized_face = cv2.resize(face, (160, 160))
        resized_face = resized_face.astype("float") / 255.0
        resized_face = np.expand_dims(resized_face, axis=0)
        preds = model.predict(resized_face)[0]
        print(preds)
        if preds > 0.5:
            label = 'spoof'
            print(label)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        else:
            label = 'real'
            print(label)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame


@app.route('/process_video', methods=['POST'])
def check_liveness():
    try:
        base64_image = request.data
        image_array = np.frombuffer(base64_image, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face = image[y-5:y+h+5, x-5:x+w+5]
            resized_face = cv2.resize(face, (160, 160))
            resized_face = resized_face.astype("float") / 255.0
            resized_face = np.expand_dims(resized_face, axis=0)
            preds = model.predict(resized_face)[0]
            print(preds)
            if preds > 0.5:
                print('spoof')
                result = {'label': 'spoof'}
                status_code = 200
            else:
                print('real')
                result = {'label': 'real'}
                status_code = 200
            return make_response(jsonify(result), status_code)
        return make_response(jsonify({'label': 'no_face_detected'}), 400)
    except Exception as e:
        print('error')
        return make_response(jsonify({'error': str(e)}), 500)


# def process_video():
#     stop_stream = False  # flag variable for stopping the stream

#     def generate():
#         with app.test_request_context():
#             nonlocal stop_stream  # use the flag variable from the outer function

#             while not stop_stream:
#                 # Read video frame
#                 data = request.data

#                 if not data:
#                     continue

#                 # Decode base64 data
#                 img_bytes = base64.b64decode(data)

#                 # Convert bytes to numpy array
#                 nparr = np.frombuffer(img_bytes, np.uint8)

#                 # Decode image
#                 img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#                 # Process frame for liveness detection
#                 processed_frame = detect_liveness(img)

#                 # Encode image as JPEG string
#                 ret, jpeg = cv2.imencode('.jpg', processed_frame)
#                 frame = jpeg.tobytes()

#                 # Yield the frame in byte format
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#     return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    data = request.get_json()
    stop = data.get('stop')

    if stop:
        global stop_stream
        stop_stream = True

    return 'OK', 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
