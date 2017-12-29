import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, request, session, flash, send_from_directory
from werkzeug.utils import secure_filename
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import StratifiedKFold

app = Flask(__name__)

# Constraints
UPLOAD_FOLDER = './static/uploaded'
ALLOWED_EXTENSIONS = set(['csv'])
RANDOM_STATE = 42

app.secret_key = 'drinking'

linreg1 = LinearRegression()
linreg2 = LinearRegression()
linreg3 = LinearRegression()
y_pred = []

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def init():
    global linreg1
    global linreg2
    global linreg3

    df = pd.read_csv(UPLOAD_FOLDER + '/student-train.csv')

    df=df.drop_duplicates(["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"])

    #Sum all the alc
    df['Dalc'] = df['Dalc'] + df['Walc']

    df.replace(['yes','no'],[1,0], inplace = True)
    df['school'] = df.loc[:,'school'].replace(['GP','MS'], [1,0])
    df['sex'] = df.loc[:,'sex'].replace(['M','F'], [1,0])
    df['address'] = df.loc[:,'address'].replace(['U','R'], [1,0])
    df['famsize'] = df.loc[:,'famsize'].replace(['LE3','GT3'], [1,0])
    df['Pstatus'] = df.loc[:,'Pstatus'].replace(['T','A'], [1,0])

    df_with_dumnies = pd.get_dummies(df,columns=['Mjob', 'Fjob', 'reason', 'guardian'])

    df_with_dumnies_2 = df_with_dumnies.loc[:,~df_with_dumnies.columns.isin(['G1', 'G2', 'G3'])]

    skf = StratifiedKFold(n_splits=5, random_state=RANDOM_STATE)

    df_with_dumnies_lin = df_with_dumnies_2.copy()

    # Dummie G1 prediction
    TARGET = 'G1'

    df_with_dumnies_lin['G1'] = 0
    y = df_with_dumnies.loc[:,TARGET]
    X = df_with_dumnies.loc[:,~df_with_dumnies.columns.isin([TARGET])]

    for train_idx, test_idx in skf.split(X, y):
        linreg1.fit(X.iloc[train_idx], y.iloc[train_idx])
        G_pred_lin = linreg1.predict(X.iloc[test_idx])
        df_with_dumnies_lin['G1'].iloc[test_idx] = G_pred_lin

    # Dummie G2 prediction
    TARGET = 'G2'

    df_with_dumnies_lin['G2'] = 0

    y2 = df_with_dumnies.loc[:,TARGET]
    X2 = df_with_dumnies_lin.loc[:,~df_with_dumnies_lin.columns.isin([TARGET])]

    for train_idx, test_idx in skf.split(X2, y2):
        linreg2.fit(X2.iloc[train_idx], y2.iloc[train_idx])
        G_pred_lin = linreg2.predict(X2.iloc[test_idx])
        df_with_dumnies_lin['G2'].iloc[test_idx] = G_pred_lin

    # Dummie G3 prediction
    TARGET = 'G3'

    y3 = df_with_dumnies.loc[:,TARGET]
    X3 = df_with_dumnies_lin

    y_pred = y3.copy()

    for train_idx, test_idx in skf.split(X3, y3):
        linreg3.fit(X3.iloc[train_idx], y3.iloc[train_idx])
        G_pred_lin = linreg3.predict(X3.iloc[test_idx])
        y_pred.iloc[test_idx] = G_pred_lin

    # Explained variance score lin:
    print('Lin Variance score: %.2f' % explained_variance_score(y3, y_pred))
    # Mean squared error regression loss lin:
    print('Lin Mean squared error regression loss: %.2f' % mean_squared_error(y3, y_pred))

    return 'Model trained', 200

@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            # First save
            file.save(UPLOAD_FOLDER + '/' + filename)
            
            df = pd.read_csv(UPLOAD_FOLDER + '/' + filename)

            skf = StratifiedKFold(n_splits=5, random_state=RANDOM_STATE)
            df=df.drop_duplicates(["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"])  

            #Preparing dataframe
            df['Dalc'] = df['Dalc'] + df['Walc']
            df.replace(['yes','no'],[1,0], inplace = True)
            df['school'] = df.loc[:,'school'].replace(['GP','MS'], [1,0])
            df['sex'] = df.loc[:,'sex'].replace(['M','F'], [1,0])
            df['address'] = df.loc[:,'address'].replace(['U','R'], [1,0])
            df['famsize'] = df.loc[:,'famsize'].replace(['LE3','GT3'], [1,0])
            df['Pstatus'] = df.loc[:,'Pstatus'].replace(['T','A'], [1,0])
            df_with_dumnies = pd.get_dummies(df,columns=['Mjob', 'Fjob', 'reason', 'guardian']) 

            df_with_dumnies_lin = df_with_dumnies.loc[:,~df_with_dumnies.columns.isin(['G1', 'G2', 'G3'])]

            df_result = pd.DataFrame()

            TARGET = 'G1'
            df_with_dumnies_lin['G1'] = None
            y = df_with_dumnies.loc[:,TARGET]
            X = df_with_dumnies.loc[:,~df_with_dumnies.columns.isin([TARGET])]

            #Predic G1
            for train_idx, test_idx in skf.split(X, y):
                G_pred_lin = linreg1.predict(X.iloc[test_idx])
                df_with_dumnies_lin['G1'].iloc[test_idx] = G_pred_lin

            df_result['G1'] = df_with_dumnies_lin['G1'].copy()

            #Predic G2
            TARGET = 'G2'

            df_with_dumnies_lin['G2'] = 0

            y2 = df_with_dumnies.loc[:,TARGET]
            X2 = df_with_dumnies_lin.loc[:,~df_with_dumnies_lin.columns.isin([TARGET])]

            for train_idx, test_idx in skf.split(X2, y2):
                G_pred_lin = linreg2.predict(X2.iloc[test_idx])
                df_with_dumnies_lin['G2'].iloc[test_idx] = G_pred_lin

            df_result['G2'] = df_with_dumnies_lin['G2'].copy()

            #Predic G3
            TARGET = 'G3'

            y3 = df_with_dumnies.loc[:,TARGET]
            X3 = df_with_dumnies_lin

            y_pred = y3.copy()

            for train_idx, test_idx in skf.split(X3, y3):
                G_pred_lin = linreg3.predict(X3.iloc[test_idx])
                y_pred.iloc[test_idx] = G_pred_lin

            df_result['G3'] = pd.Series(y_pred)

            print(df_result)

            # POST OK
            return render_template('result.html', result=df_result.as_matrix())

        else: # Error
            # POST ERROR
            return render_template('upload.html', error="Cannot upload this file")

    # GET
    return render_template('upload.html', error=None)

if __name__ == '__main__':
    port = int(os.getenv("PORT"))
    debug = os.getenv("FLASK_DEBUG")
    app.run(host='0.0.0.0', port=port, debug=debug)