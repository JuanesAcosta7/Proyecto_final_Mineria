from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

@app.route('/')
def inicio():
    return render_template('index.html')

@app.route('/etapa1')
def etapa1():
    return render_template('etapa1.html')

@app.route('/etapa2')
def etapa2():
    return render_template('etapa2.html')

@app.route('/etapa3')
def etapa3():

    df = pd.read_csv(
        'data/reglas.csv',
        sep=';'
    )

    tabla = df.to_html(
        classes='table',
        index=False
    )

    return render_template(
        'etapa3.html',
        tabla_reglas=tabla
    )

@app.route('/etapa4')
def etapa4():
    return render_template('etapa4.html')

if __name__ == '__main__':
    app.run(debug=True)