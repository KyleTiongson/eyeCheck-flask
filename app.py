import os
from flask import Flask, request, jsonify, send_file, Response, render_template_string
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import cv2
import tempfile
import base64
import logging
from tensorflow.lite.python.interpreter import Interpreter

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG)


# Load labels from a file
def load_labels(lblpath):
    with open(lblpath, 'r') as f:
        return [line.strip() for line in f.readlines()]


# Preprocess image for the model
def preprocess_image(image, input_shape):
    image = Image.open(io.BytesIO(image))
    image = image.convert('RGB')
    image_resized = image.resize((input_shape[1], input_shape[2]))
    image_np = np.array(image_resized)
    return image_np


# Perform detection using TFLite model
def tflite_detect_image(modelpath, lblpath, image_data, min_conf=0.1):
    labels = load_labels(lblpath)
    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']

    image_np = preprocess_image(image_data, input_shape)
    input_data = np.expand_dims(image_np, axis=0).astype(np.float32)
    input_data = (input_data - 127.5) / 127.5  # Normalize image

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[3]['index'])[0]
    scores = interpreter.get_tensor(output_details[0]['index'])[0]

    imH, imW, _ = image_np.shape
    detections = []
    for i in range(len(scores)):
        if scores[i] > min_conf:
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))
            object_name = labels[int(classes[i])]
            detections.append({
                "object": object_name,
                "confidence": float(scores[i]),
                "box": [xmin, ymin, xmax, ymax]
            })

    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

    # Handle one or two eye detections
    if len(detections) == 2:
        if detections[0]['box'][0] < detections[1]['box'][0]:
            detections[0]['object'] = 'left_eye'
            detections[1]['object'] = 'right_eye'
        else:
            detections[0]['object'] = 'right_eye'
            detections[1]['object'] = 'left_eye'
    elif len(detections) == 1:
        if detections[0]['box'][0] < imW / 2:
            detections[0]['object'] = 'left_eye'
        else:
            detections[0]['object'] = 'right_eye'

    for detection in detections:
        xmin, ymin, xmax, ymax = detection['box']
        cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
        label = f"{detection['object']}: {int(detection['confidence'] * 100)}%"
        cv2.putText(image_np, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file_path = temp_file.name
    logging.debug(f"Saving detected image to {temp_file_path}")
    cv2.imwrite(temp_file_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    return detections, temp_file_path


@app.route('/detect', methods=['POST'])
def detect():
    logging.debug("Received request for /detect endpoint")
    data = request.get_json()
    if 'image' not in data:
        logging.error("No image part in the request")
        return jsonify(error="No image part"), 400

    image_data = base64.b64decode(data['image'])

    if not image_data:
        logging.error("No selected image")
        return jsonify(error="No selected image"), 400

    model_path = 'detect4.tflite'
    label_path = 'labelmap.txt'
    min_conf_threshold = 0.5  # Lower threshold for testing

    detections, temp_image_path = tflite_detect_image(model_path, label_path, image_data, min_conf_threshold)

    response = {
        'detections': detections,
        'image_url': f'http://{request.host}/image/{os.path.basename(temp_image_path)}'
    }

    logging.debug(f"Response: {response}")
    return jsonify(response)


@app.route('/image/<filename>')
def get_image(filename):
    logging.debug(f"Received request for image {filename}")
    file_path = os.path.join(tempfile.gettempdir(), filename)
    logging.debug(f"File path: {file_path}")

    # Check if the file exists
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return jsonify(error="File not found"), 404

    response = send_file(file_path, mimetype='image/jpeg')

    # Cleanup the temporary file after sending the response
    os.remove(file_path)

    return response


@app.route('/')
def index():
    return render_template_string("""
       <!doctype html>
       <title>Flask App</title>
       <h1>Flask App is running</h1>
       <p>To use the detection endpoint, send a POST request to <code>/detect</code> with your image data.</p>
       """)


if __name__ == '__main__':
    logging.debug("Starting Flask app")
    app.run()
