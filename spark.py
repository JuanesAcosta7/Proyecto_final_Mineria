import findspark
findspark.init()

import os
import sys
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import *

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = SparkSession.builder \
    .appName("CapacidadHospitalariaCluster") \
    .master("spark://192.168.80.17:7077") \
    .config("spark.driver.host", "192.168.80.17") \
    .config("spark.driver.bindAddress", "0.0.0.0") \
    .config(
        "spark.jars",
        "jdbc/mssql-jdbc-13.4.0.jre11.jar"
    ) \
    .config(
        "spark.executor.memory",
        "2g"
    ) \
    .config(
        "spark.driver.memory",
        "2g"
    ) \
    .config(
        "spark.executor.cores",
        "2"
    ) \
    .config(
        "spark.cores.max",
        "4"
    ) \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

url = "jdbc:sqlserver://localhost:1433;databaseName=CAPACIDADHOSPITALARIA;integratedSecurity=true;trustServerCertificate=true"

properties = {
    "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}

inicio_carga = time.time()

registros = spark.read.jdbc(
    url=url,
    table="Registros",
    properties=properties
)

departamentos = spark.read.jdbc(
    url=url,
    table="Departamentos",
    properties=properties
)

regiones = spark.read.jdbc(
    url=url,
    table="Regiones",
    properties=properties
)

naturalezas = spark.read.jdbc(
    url=url,
    table="Naturalezas_Juridicas",
    properties=properties
)

tipos = spark.read.jdbc(
    url=url,
    table="Tipos_Capacidad",
    properties=properties
)

categorias = spark.read.jdbc(
    url=url,
    table="Categorias",
    properties=properties
)

fin_carga = time.time()

print("\nTIEMPO DE CARGA")
print(fin_carga - inicio_carga)

print("\nREGISTROS")
print(registros.count())

df = registros \
    .join(
        departamentos,
        registros.Id_Departamento == departamentos.Id_Departamento
    ) \
    .join(
        regiones,
        departamentos.Id_Region == regiones.Id_Region
    ) \
    .join(
        naturalezas,
        registros.Id_Naturaleza == naturalezas.Id_Naturaleza
    ) \
    .join(
        tipos,
        registros.Id_Tipo == tipos.Id_Tipo
    ) \
    .join(
        categorias,
        tipos.Id_Categoria == categorias.Id_Categoria
    )

df = df.select(
    regiones.Nombre.alias("Region"),
    departamentos.Nombre.alias("Departamento"),
    naturalezas.Nombre.alias("Naturaleza"),
    tipos.Nombre.alias("Tipo_Capacidad"),
    categorias.Nombre.alias("Categoria"),
    registros.Ano.alias("Ano"),
    registros.Cantidad.alias("Cantidad")
)

df = df.dropna()

df = df.filter(col("Cantidad") > 0)

df = df.repartition(4)

print("\nDATAFRAME MAESTRO")
df.show(10)

print("\nESQUEMA")
df.printSchema()

print("\nCAPACIDAD POR REGION")

capacidad_region = df.groupBy(
    "Region"
).agg(
    sum("Cantidad").alias("Total_Capacidad")
).orderBy(
    desc("Total_Capacidad")
)

capacidad_region.show()

print("\nCAPACIDAD POR DEPARTAMENTO")

capacidad_departamento = df.groupBy(
    "Departamento"
).agg(
    sum("Cantidad").alias("Total_Capacidad")
).orderBy(
    desc("Total_Capacidad")
)

capacidad_departamento.show()

print("\nTIPOS DE CAPACIDAD MAS IMPORTANTES")

tipos_importantes = df.groupBy(
    "Tipo_Capacidad"
).agg(
    sum("Cantidad").alias("Total")
).orderBy(
    desc("Total")
)

tipos_importantes.show()

print("\nINDICADOR DE COMPLEJIDAD")

df_complejidad = df.withColumn(
    "Peso",
    when(col("Tipo_Capacidad").like("%UCI%"), 5)
    .when(col("Tipo_Capacidad").like("%Quir%"), 4)
    .when(col("Tipo_Capacidad").like("%Hospital%"), 3)
    .when(col("Tipo_Capacidad").like("%Ambulancia%"), 2)
    .otherwise(1)
)

df_complejidad = df_complejidad.withColumn(
    "Indice_Complejidad",
    col("Cantidad") * col("Peso")
)

ranking = df_complejidad.groupBy(
    "Departamento"
).agg(
    sum("Indice_Complejidad").alias("Complejidad_Total")
).orderBy(
    desc("Complejidad_Total")
)

ranking.show()

print("\nPREPARACION MACHINE LEARNING")

ml = df_complejidad.groupBy(
    "Departamento"
).agg(
    sum("Cantidad").alias("Cantidad_Total"),
    sum("Indice_Complejidad").alias("Complejidad")
)

assembler = VectorAssembler(
    inputCols=[
        "Cantidad_Total",
        "Complejidad"
    ],
    outputCol="features"
)

ml_data = assembler.transform(ml)

print("\nKMEANS")

kmeans = KMeans(
    k=3,
    seed=1
)

modelo = kmeans.fit(ml_data)

resultado = modelo.transform(ml_data)

resultado.select(
    "Departamento",
    "Cantidad_Total",
    "Complejidad",
    "prediction"
).show()

print("\nANOMALIAS HOSPITALARIAS")

promedio = ml.agg(
    avg("Complejidad").alias("Promedio")
).collect()[0]["Promedio"]

anomalias = ml.withColumn(
    "Diferencia",
    abs(col("Complejidad") - promedio)
)

anomalias = anomalias.orderBy(
    desc("Diferencia")
)

anomalias.show(20)

print("\nPREDICCION DE CRECIMIENTO")

prediccion_df = df_complejidad.groupBy(
    "Ano",
    "Departamento"
).agg(
    sum("Indice_Complejidad").alias("Complejidad_Total")
)

assembler_pred = VectorAssembler(
    inputCols=["Ano"],
    outputCol="features"
)

prediccion_df = assembler_pred.transform(prediccion_df)

prediccion_df = prediccion_df.withColumnRenamed(
    "Complejidad_Total",
    "label"
)

train, test = prediccion_df.randomSplit(
    [0.8, 0.2],
    seed=1
)

lr = LinearRegression()

modelo_lr = lr.fit(train)

predicciones = modelo_lr.transform(test)

print("\nRESULTADOS REGRESION LINEAL")

predicciones.select(
    "Departamento",
    "Ano",
    "label",
    "prediction"
).show(20)

evaluador = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="r2"
)

r2 = evaluador.evaluate(predicciones)

print("\nR2 DEL MODELO")
print(r2)

os.makedirs("resultados", exist_ok=True)

print("\nEXPORTANDO RESULTADOS")

capacidad_region.toPandas().to_csv(
    "resultados/capacidad_region.csv",
    index=False
)

capacidad_departamento.toPandas().to_csv(
    "resultados/capacidad_departamento.csv",
    index=False
)

tipos_importantes.toPandas().to_csv(
    "resultados/tipos_importantes.csv",
    index=False
)

ranking.toPandas().to_csv(
    "resultados/ranking_complejidad.csv",
    index=False
)

resultado.select(
    "Departamento",
    "Cantidad_Total",
    "Complejidad",
    "prediction"
).toPandas().to_csv(
    "resultados/clusters.csv",
    index=False
)

anomalias.toPandas().to_csv(
    "resultados/anomalias.csv",
    index=False
)

predicciones.select(
    "Departamento",
    "Ano",
    "label",
    "prediction"
).toPandas().to_csv(
    "resultados/predicciones.csv",
    index=False
)

print("\nINFORMACION DEL CLUSTER")

print("\nMASTER")
print(spark.sparkContext.master)

print("\nPARTICIONES")
print(df.rdd.getNumPartitions())

print("\nPROCESAMIENTO DISTRIBUIDO ACTIVO")

print("\nPROCESO COMPLETADO")

spark.stop()