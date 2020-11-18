import time
from absl import app, logging
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
from flask import Flask, request, Response, jsonify, send_from_directory, abort
import os


####################################
# USAGE
# python test_handwriting.py --model handwriting.model --image images/umbc_address.png

# import the necessary packages
from tensorflow.keras.models import load_model
from imutils.contours import sort_contours
import numpy as np
import argparse
import imutils
import cv2
################################
import math
import operator # for sorting li


def roundup(x): # round to nearest 10
    return int(math.ceil(x / 10.0)) * 10
#################################


# load the handwriting OCR model

print("[INFO] loading handwriting OCR model...")
model = load_model("C:/repos/env/Text-Detection/handwriting.model")
######################################################
# customize your API through the following parameters
classes_path = './data/labels/coco.names'
weights_path = './weights/yolov3.tf'
tiny = False                    # set to True if using a Yolov3 Tiny model
size = 416                      # size images are resized to for model
output_path = './detections/'   # path to output folder where images with detections are saved
num_classes = 7                # number of classes in model

# load in weights and classes
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

if tiny:
    yolo = YoloV3Tiny(classes=num_classes)
else:
    yolo = YoloV3(classes=num_classes)

yolo.load_weights(weights_path).expect_partial()
print('weights loaded')

class_names = [c.strip() for c in open(classes_path).readlines()]
print('classes loaded')

# Initialize Flask application
app = Flask(__name__)


class finalList:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y


# Initialize Flask application
app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello_world():
    return '''<html>
<body>
<p>Click on the "Choose File" button to upload a file:</p>
<form action="/detections"  method = "POST" 
        enctype = "multipart/form-data">
  <input type="file" id="images" name="images">
  <input type="submit" value="Upload Image" name="submit">
</form>
</body>
</html>'''


# API that returns JSON with classes found in images
@app.route('/detections', methods=['POST'])
def get_detections():
    raw_images = []
    images = request.files.getlist("images")
    image_names = []
    for image in images:
        image_name = image.filename
        image_names.append(image_name)
        image.save(os.path.join(os.getcwd(), image_name))
        img_raw = tf.image.decode_image(
            open(image_name, 'rb').read(), channels=3)
        raw_images.append(img_raw)

    num = 0

    # create list for final response
    response = []
    li = []
    for j in range(len(raw_images)):
        # create list of responses for current image
        responses = []
        raw_img = raw_images[j]
        num += 1
        img = tf.expand_dims(raw_img, 0)
        img = transform_images(img, size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        t2 = time.time()
        print('time: {}'.format(t2 - t1))
        # print("**",scores)
        print('detections:')
        for i in range(nums[0]):
            # if np.array(scores[0][i])*100>30:
            print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                        np.array(scores[0][i]),
                                        np.array(boxes[0][i])))
            responses.append({
                "class": class_names[int(classes[0][i])],
                "confidence": float("{0:.2f}".format(np.array(scores[0][i]) * 100)),
                "co ordinates": str("{}".format((np.array(boxes[0][i]))))
            })

            # print(tuple(np.array(boxes[0][i])))

            # img = Image.open("C:\\Repos\\object-Detection-API\\detections\\detection.jpg")
            # a,b = img.size
            # print("*****")
            # print(a,b)
            x, y, z, h = np.array(boxes[0][i])
            p = finalList(class_names[int(classes[0][i])], x, y)
            li.append(p)
            # print(x,y,z,h)
            # crop = img.crop((x*a,y*b,z*a,h*b))
            # crop.show()
        response.append({
            "image": image_names[j],
            "detections": responses
        })
        # note the tuple
    img = cv2.cvtColor(raw_img.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    cv2.imwrite(output_path + 'detection' + '.jpg', img)
    print('output saved to: {}'.format(output_path + 'detection' + str(num) + '.jpg'))

    st = """
    <!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width" />
<title>HTML Result</title>
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.1/css/bootstrap.min.css" 
      integrity="sha384-WskhaSGFgHYWDcbwN70/dfYBj47jz9qbsMId/iRN3ewGhXQFZCSftd1LZCfmhktB" crossorigin="anonymous">
</head>
<body>
<div class="container body-content">"""

    en = """
</div>
</body>
</html>
    """
    inputf = """ 
    <div class="row justify-content-start" style="padding-top:10px;">
<label>Demo Text: </label>
</div>
<div class="row justify-content-center" style="padding-top:10px;">
<input class="form-control"></input>
            </div>"""

    button = """
    <div class="col" style="padding-top:10px;">
        <button class="btn btn-primary">Submit</button>
    </div>"""

    img = """
    <img src="C:/repos/env/Object-Detection-API/img.png" width="150" height="150" alt="Image Here">"""

    radio = """
    <div class="col" style="padding-top:10px;">
        <input type="radio" id="male" name="Demo text" value="male">
        <label for="male">Demo Text</label><br>
    </div>
"""
    dropdown = """
    <div class="dropdown">
<label for="cars">Dropdown:</label>
<select name="cars" id="cars" class="btn btn-primary dropdown-toggle">
<option value="1">Option 1</option>
<option value="2">Option 2</option>
<option value="3">Option 3</option>
<option value="4">Option 4</option>
</select>
</div>"""
    checkbox = """
    <div class="col" style="padding-top:10px;">
        <input type="checkbox" id="vehicle1" name="vehicle1" value="Bike">
        <label for="vehicle1"> I have a bike</label><br>
    </div>
    """
    text = """<div class="col" style="padding-top:10px;"> <p class="text-black-50"> You’ve probably heard of 
    Lorem Ipsum before – it’s the most-used dummy text excerpt out there. People use it because it has a fairly 
    normal distribution of letters and words (making it look like normal English), but it’s also Latin, 
    which means your average reader won’t get distracted by trying to read it. </p> </div> """

    sorted_li = sorted(li, key=operator.attrgetter('y'))
    # print("###########################")
    # for m in sorted_li:
    #     print(m.name, m.y)
    #
    # print("###########################")
    for i in sorted_li:
        if i.name == "check box":
            st += checkbox
        elif i.name == "radio button":
            st += radio
        elif i.name == "dropdown":
            st += dropdown
        elif i.name == "input":
            st += inputf
        elif i.name == "submit":
            st += button
        elif i.name == "text":
            st += text
        else:
            st += img
        print(i.name, i.x, i.y)
    print(st + en)
    f = open("demofile3.html", "w")
    f.write(st + en)
    f.close()
    # remove temporary images
    for name in image_names:
        os.remove(name)
    try:
        return jsonify({"response": response}), 200
    except FileNotFoundError:
        abort(404)


# API that returns image with detections on it
# @app.route('/image', methods= ['POST'])
# def get_image():
#     image = request.files["images"]
#     image_name = image.filename
#     image.save(os.path.join(os.getcwd(), image_name))
#     img_raw = tf.image.decode_image(
#         open(image_name, 'rb').read(), channels=3)
#     img = tf.expand_dims(img_raw, 0)
#     img = transform_images(img, size)

#     t1 = time.time()
#     boxes, scores, classes, nums = yolo(img)
#     t2 = time.time()
#     print('time: {}'.format(t2 - t1))

#     print('detections:')
#     for i in range(nums[0]):
#         print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
#                                         np.array(scores[0][i]),
#                                         np.array(boxes[0][i])))
#     img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
#     img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
#     cv2.imwrite(output_path + 'detection.jpg', img)
#     print('output saved to: {}'.format(output_path + 'detection.jpg'))

#     # prepare image for response
#     _, img_encoded = cv2.imencode('.png', img)
#     response = img_encoded.tostring()

#     #remove temporary image
#     os.remove(image_name)

#     try:
#         return Response(response=response, status=200, mimetype='image/png')
#     except FileNotFoundError:
#         abort(404)

# API that returns image with detections on it


@app.route('/image', methods=['POST'])
def get_image():
    image = request.files["images"]
    # print("######### IMG", image)
    image_name = image.filename
    image.save(os.path.join(os.getcwd(), image_name))
    img_raw = tf.image.decode_image(
        open(image_name, 'rb').read(), channels=3)
    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, size)

    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
    print('time: {}'.format(t2 - t1))

    print('detections:')
    for i in range(nums[0]):
        print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                    np.array(scores[0][i]),
                                    np.array(boxes[0][i])))
    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    cv2.imwrite(output_path + 'detection.jpg', img)
    print('output saved to: {}'.format(output_path + 'detection.jpg'))

    # prepare image for response
    _, img_encoded = cv2.imencode('.png', img)
    response = img_encoded.tostring()

    ######################################################################
    image_path = os.path.join(os.getcwd(), 'detections/detection.jpg')

    image = cv2.imread(image_path)
    print(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # perform edge detection, find contours in the edge map, and sort the
    # resulting contours from left-to-right
    edged = cv2.Canny(blurred, 30, 150)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]

    # initialize the list of contour bounding boxes and associated
    # characters that we'll be OCR'ing
    chars = []

    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # filter out bounding boxes, ensuring they are neither too small
        # nor too large
        if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
            # extract the character and threshold it to make the character
            # appear as white (foreground) on a black background, then
            # grab the width and height of the thresholded image
            roi = gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255,
                                   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            (tH, tW) = thresh.shape

            # if the width is greater than the height, resize along the
            # width dimension
            if tW > tH:
                thresh = imutils.resize(thresh, width=32)

            # otherwise, resize along the height
            else:
                thresh = imutils.resize(thresh, height=32)

            # re-grab the image dimensions (now that its been resized)
            # and then determine how much we need to pad the width and
            # height such that our image will be 32x32
            (tH, tW) = thresh.shape
            dX = int(max(0, 32 - tW) / 2.0)
            dY = int(max(0, 32 - tH) / 2.0)

            # pad the image and force 32x32 dimensions
            padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                                        left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0))
            padded = cv2.resize(padded, (32, 32))

            # prepare the padded image for classification via our
            # handwriting OCR model
            padded = padded.astype("float32") / 255.0
            padded = np.expand_dims(padded, axis=-1)

            # update our list of characters that will be OCR'd
            chars.append((padded, (x, y, w, h)))

    # extract the bounding box locations and padded characters
    boxes = [b[1] for b in chars]
    chars = np.array([c[0] for c in chars], dtype="float32")

    # OCR the characters using our handwriting recognition model
    preds = model.predict(chars)

    # define the list of label names
    labelNames = "0123456789"
    labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    labelNames = [l for l in labelNames]
    lst = []
    # loop over the predictions and bounding box locations together
    for (pred, (x, y, w, h)) in zip(preds, boxes):
        # find the index of the label with the largest corresponding
        # probability, then extract the probability and label
        i = np.argmax(pred)
        prob = pred[i]
        label = labelNames[i]
        lst.append(label + ":" + str(prob))
        # draw the prediction on the image
        print("[INFO] {} - {:.2f}%".format(label, prob * 100))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # # show the image
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)

    # remove temporary image
    os.remove(image_name)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

    try:
        return jsonify({"response": lst}), 200
    except FileNotFoundError:
        abort(404)


if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port=5000)