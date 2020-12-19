from flask import Flask, request, render_template, jsonify, redirect, url_for, send_from_directory
from imageai.Detection import ObjectDetection
from keras import backend as k
import os
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename


app = Flask(__name__)

@app.route('/')
def home_page():
	# print(request.args)
	if request.args.get('detected')=='true':
		return render_template('index.html', detected="true", new=request.args.get('new'), old=request.args.get('old'))
	else:
		return render_template('index.html', detected="false", new=request.args.get('new'), old=request.args.get('old'))

@app.route('/detect', methods = ['POST'])
def det():
	print(request.files)
	img = request.files['file']
	imgName = secure_filename(img.filename)
	newName = "new_"+imgName
	img.save(os.path.join(os.getcwd() + '\\static\\uploads', imgName))
	# print(type(Image.open(request.files['file'])))
	# print(type(np.frombuffer(img, dtype=np.uint8)))
	# print(type(np.asarray(img, dtype='float64')))
	detect(img, imgName, newName)
	return redirect(url_for('home_page', detected="true", old=imgName, new=newName, **request.args))
	# return render_template('result.html', im=imgName)


def detect(img, fName, newName):
	k.clear_session()
	exe_path = os.getcwd()
	source_path = exe_path + '\\static\\uploads'
	

	det = ObjectDetection()
	det.setModelTypeAsRetinaNet()
	det.setModelPath( os.path.join(exe_path, "resnet50_coco_best_v2.1.0.h5"))

	det.loadModel()
	detections = det.detectObjectsFromImage(
		# input_image= os.path.join(source_path, fName),
		input_image= img, 
		output_image_path=os.path.join(source_path, newName))

if __name__ == '__main__':
    app.run()