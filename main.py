import os
import time
import pandas as pd
import numpy as np
from flask import *
import pickle
import sklearn
from pandas.core.dtypes.common import classes

ALLOWED_EXTENSIONS = set(['csv','xls','xlsx'])
app = Flask(__name__)

clf = pickle.load(open('fraud.pkl', 'rb'))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def upload():
    return render_template("index.html")


@app.route('/success', methods=['POST','GET'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        if f and allowed_file(f.filename):
            f.save("files/"+f.filename)
            return render_template("success.html", data=Markup(data("files/"+f.filename)))
    #time.sleep(200)

    return render_template("failed.html")

@app.route('/download')
def downloadFile ():
    path = r'download_file/otchet.csv'
    return send_file(path, as_attachment=True)

def data(file):
    file_data = pd.read_csv(file)
    df=pd.DataFrame(file_data)
    if 'Class' in df.columns:
        df=df.drop(['Class'], axis = 1)
    if 'Time' in df.columns:
        df_result=np.asarray(df.drop(['Time'], axis = 1))
    else:
        df_result = np.asarray(df)
    prediction=clf.predict(df_result)
    df['Pred'] = prediction
    res=df[df['Pred']==1]
    html = res.to_html()
    res.to_csv("download_file/otchet.csv", sep='\t', encoding='utf-8')
    os.remove(file)
    return html



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4567)

