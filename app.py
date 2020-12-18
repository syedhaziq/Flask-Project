from flask import Flask, render_template,request
import pickle
import numpy as np

with open('model.pickle', 'rb') as f:
    clf = pickle.load(f)

app = Flask(__name__)


@app.route("/")
def about():
    
    return render_template('irisabout.HTML')

@app.route("/predict", methods = ['GET', 'POST'])
def predict():
    if request.method == "POST":
        sepal_length1= request.form.get('sepal_length')
        sepal_width1=request.form.get('sepal_width')
        petal_length1=request.form.get('petal_length')
        petal_width1 = request.form.get('petal_width')
        
        values=np.array([[sepal_length1,sepal_width1,petal_length1,petal_width1]])
        pred=clf.predict(values)[0]
        if pred==0:
            name='setosa'
        elif pred == 1:
            
            name='versicolor'
        else:
            name="virginica"
                
                         
        return render_template('irispredict.HTML', pred=pred, name=name)
    
                           
    pass                     
                           
app.run(debug=True)                         
                           
                               