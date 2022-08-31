from flask import Flask,render_template,request
import pickle
import numpy as np
#create a flask app
app=Flask(__name__)
#load the model
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def hello():
    return render_template('home.html')

@app.route('/check',methods=['POST','GET'])
def red():
    return render_template('index.html')

@app.route('/titanic',methods=['POST','GET'])
def predict():
    if request.method=="POST":
        x=[int(i) for i in request.form.values()]
    final=[x]
    #print(x) 
    #print(final)
    ans=model.predict(final)
    #print(ans)
    if ans[0]==1:
        return render_template('survival.html')
    else:
        return render_template('death.html')
    
if __name__=='__main__':
    app.run()


