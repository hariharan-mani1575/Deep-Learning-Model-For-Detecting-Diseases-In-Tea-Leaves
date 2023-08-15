from flask import Flask,render_template,redirect,request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename

#MODEL = tf.keras.models.load_model("E:/NaanMudhalvan/model_vgg16.h5")

MODEL = tf.keras.models.load_model("D:/NM_PROJECT/project_final/tea.hdf5")

CLASS_NAMES = ["algal leaf","Anthracnose","bird eye spot","brown blight","gray light","healthy","red leaf spot","white spot"]

app = Flask(__name__,static_folder="static/")
def model_predict(img_path, model):

    img = image.load_img(img_path,target_size=(180,180))

    # Preprocessing the image
    img_arr = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x = np.expand_dims(img_arr, axis=0)
    #print(resized.shape)
    #ip = np.array()
    preds = model.predict(x)
    return preds

@app.route('/',methods=['GET'])
def index():
    return render_template('Home.html')

@app.route('/About.html')
def about():
    return render_template("About.html")

@app.route('/Home.html')
def home():
    return render_template('Home.html')

@app.route('/Contact.html')
def leaf():
    return render_template("Contact.html")

@app.route('/dtes.html')
def dtes():
    return render_template('dtes.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
     if request.method=='POST':
        f=request.files['file']
        
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
           basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

      #image = np.array(Image.open(f))
       # img_batch = np.expand_dims(image, 0)
       # predictions = MODEL.predict(img_batch)''''
        predictions = model_predict(file_path, MODEL)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        return predicted_class
            #"confidence": float(confidence)
       # }


if __name__=="__main__":
    app.run(debug=True)