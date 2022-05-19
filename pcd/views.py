import cv2
from flask import request, render_template, redirect

from pcd import app
from pcd.utils import *


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)

        img_file = request.files['image']

        if img_file.filename == '':
            return redirect(request.url)

        if img_file and allowed_file(img_file.filename):
            img_str = img_file.read()
            image = decode_image(img_str)
            corners = predict(image)
            image = draw_corners(image, corners)
            image = cv2.resize(image, (480, 720))
            img_str = encode_image(image)
            return render_template('index.html', img_data=img_str)
    return render_template('base.html')