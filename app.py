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

    capacidad_region = pd.read_csv(
        'resultados/capacidad_region.csv'
    )

    capacidad_departamento = pd.read_csv(
        'resultados/capacidad_departamento.csv'
    )

    clustering = pd.read_csv(
        'resultados/clusters.csv'
    )

    anomalias = pd.read_csv(
        'resultados/anomalias.csv'
    )

    tabla_region = capacidad_region.to_html(
        classes='table',
        index=False
    )

    tabla_departamento = capacidad_departamento.to_html(
        classes='table',
        index=False
    )

    tabla_clustering = clustering.to_html(
        classes='table',
        index=False
    )

    tabla_anomalias = anomalias.to_html(
        classes='table',
        index=False
    )

    return render_template(
        'etapa4.html',
        tabla_region=tabla_region,
        tabla_departamento=tabla_departamento,
        tabla_cluster=tabla_clustering,
        tabla_anomalias=tabla_anomalias
    )

if __name__ == '__main__':
    app.run(debug=True)