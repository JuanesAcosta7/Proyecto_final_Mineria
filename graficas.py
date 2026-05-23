import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("graficas", exist_ok=True)

df_region = pd.read_csv("resultados/capacidad_region.csv")

plt.figure(figsize=(10,6))

plt.bar(
    df_region["Region"],
    df_region["Total_Capacidad"]
)

plt.xticks(rotation=45)

plt.title("Capacidad Hospitalaria por Region")
plt.ylabel("Capacidad")

plt.tight_layout()

plt.savefig("graficas/capacidad_region.png")

plt.close()

df_dep = pd.read_csv("resultados/capacidad_departamento.csv")

top10 = df_dep.head(10)

plt.figure(figsize=(12,6))

plt.bar(
    top10["Departamento"],
    top10["Total_Capacidad"]
)

plt.xticks(rotation=45)

plt.title("Top 10 Departamentos")
plt.ylabel("Capacidad")

plt.tight_layout()

plt.savefig("graficas/top_departamentos.png")

plt.close()

df_tipo = pd.read_csv("resultados/tipos_importantes.csv")

top10_tipo = df_tipo.head(10)

plt.figure(figsize=(12,6))

plt.bar(
    top10_tipo["Tipo_Capacidad"],
    top10_tipo["Total"]
)

plt.xticks(rotation=45)

plt.title("Tipos de Capacidad Mas Importantes")
plt.ylabel("Total")

plt.tight_layout()

plt.savefig("graficas/tipos_capacidad.png")

plt.close()

df_cluster = pd.read_csv("resultados/clusters.csv")

plt.figure(figsize=(10,6))

plt.scatter(
    df_cluster["Cantidad_Total"],
    df_cluster["Complejidad"],
    c=df_cluster["prediction"]
)

plt.xlabel("Cantidad Total")
plt.ylabel("Complejidad")

plt.title("Clustering KMeans")

plt.tight_layout()

plt.savefig("graficas/clustering.png")

plt.close()

df_anomalias = pd.read_csv("resultados/anomalias.csv")

plt.figure(figsize=(10,6))

plt.scatter(
    df_anomalias["Cantidad_Total"],
    df_anomalias["Complejidad"],
    c=df_anomalias["Diferencia"]
)

plt.xlabel("Cantidad Total")
plt.ylabel("Complejidad")

plt.title("Deteccion de Anomalias")

plt.tight_layout()

plt.savefig("graficas/anomalias.png")

plt.close()

df_prediccion = pd.read_csv("resultados/predicciones.csv")

plt.figure(figsize=(12,6))

plt.plot(
    df_prediccion["Ano"],
    df_prediccion["prediction"]
)

plt.xlabel("Ano")
plt.ylabel("Capacidad Predicha")

plt.title("Prediccion Temporal UCI Adulto")

plt.tight_layout()

plt.savefig("graficas/prediccion_uci.png")

plt.close()

print("GRAFICAS GENERADAS")