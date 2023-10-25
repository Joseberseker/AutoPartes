#Importamos las librerías necesarias 
import pandas as pd
import numpy as np
import datetime
import joblib
from sklearn import preprocessing
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


#Cargamos nuestro dataset
df = pd.read_csv("dataset_partes_motos.csv", delimiter=',')

#visualizamos una parte del dataframe
df.head(10)

#vefificamos si existen valores nulos
df.isnull().sum()

#Reemplazamos los valores del tipo NaN en la columna Nombre por la palabra Cilindro
df.Nombre.replace(np.nan,'Cilindros', inplace = True)
df.head(10)

#vefificamos si existen valores nulos (Podemos observar que la columna Nombre se han reemplazado
# los valores NaN por el valor Cilindros)
df.isnull().sum()


#Ahora procedemos a reemplazar los valores vacios en la columna Marca, por el valor Akt
df.Marca.replace(np.nan,'Akt', inplace = True)

#Procedemos a verificar la columna Marca si existen valores vacios
df.isnull().sum() 

#Ahora procedemos a reemplazar los valores vacios en la columna Marca, por el valor Akt
df.Proveedor.replace(np.nan,'Chaoyang', inplace = True)

#Procedemos a verificar la columna Marca si existen valores vacios
df.isnull().sum() 

#Reemplazamos los valores NaN por ceros en la columna Cantidad
df.Cantidad.replace(np.nan,0, inplace = True)
df.head(10)

#Procedemos a verificar la columna Cantidad si existen valores vacios
df.isnull().sum()

#Podemos observar los tipos de datos de cada columna
df.dtypes


#Procedemos a quitarle las comas a los valores de la columna Precio_venta
df["Precio_venta"] = pd.to_numeric(df["Precio_venta"].str.replace('$', ''))
df.Precio_venta.isnull().sum()

df.head(10)

#Como podemos observar, pudimos cambiar el tipo de dato del campo Precio_venta
df.dtypes

#Reemplazamos los valores 0 por 133000 en la columna Precio_venta
df.Precio_venta.replace(np.nan,133000, inplace = True)
df.head(10)

df.dtypes

#Procedemos a verificar si en todas las columnas existen valores vacios
df.isnull().sum()

#Convertimos la columna Mes a tipo fecha y le cambiamos el nombre a Fecha_venta. 

df['Mes'] = pd.to_datetime(df['Mes'])
df = df.rename(columns={'Mes':'Fecha_venta'})
df.head(5)

df.dtypes

#Luego, Dividimos la columna fecha, en dìas, meses y años en columnas separadas

df['Año'] = df['Fecha_venta'].dt.year
df['Mes'] = df['Fecha_venta'].dt.month
df['Dia'] = df['Fecha_venta'].dt.day

df.head(5)

#Cantidad máxima de componentes vendidos
df['Cantidad'].max()

#Promedio de componentes vendidos
df['Cantidad'].mean()

#Convertimos las columnas de Precio_venta y Cantidad al tipo entero
df['Precio_venta'] = df['Precio_venta'].astype(int)
df['Cantidad'] = df['Cantidad'].astype(int)

#Crear una columna total
df['Total'] = df['Precio_venta'] * df['Cantidad']
df.head(5)

df.dtypes

#Creamos una columna condicional llamada Compra_distribuidor que 
#tendrá tres opciones, Encargar_Mas, Encargar Normal y Encargar_Menos

condiciones =[
    (df['Cantidad'] <= 300),
    (df['Cantidad'] >= 300) & (df['Cantidad'] <= 487),
    (df['Cantidad'] > 487)
]

opciones = ['Encargar_Menos','Encargar_Normal', 'Encargar_Mas']
df['Decision'] = np.select(condiciones, opciones)
df['Decision'].value_counts()

df.head(5)



#Algoritmo de DecisionTreeClassifier

#Cuando llamamos a fit_transform () en el conjunto de entrenamiento 
#para realizar el primer ajuste y luego el procesamiento estandarizado
# de los datos de entrenamiento
decision = preprocessing.LabelEncoder()
df['Decision'] = decision.fit_transform(df['Decision'] )
nombre = preprocessing.LabelEncoder()
df['Nombre'] = nombre.fit_transform(df['Nombre'] )
marca = preprocessing.LabelEncoder()
df['Marca'] = marca.fit_transform(df['Marca'] )
proveedor = preprocessing.LabelEncoder()
df['Proveedor'] = proveedor.fit_transform(df['Proveedor'] )


#Reemplazamos las categorías de la columna Decicion ("Encargar_Menos","Encargar_Normal","Encargar_Mas") 
#Por los valores 0,1 y 2 respectivamente

list(decision.inverse_transform([0,1,2]))

#3)	Reemplazamos las categorías de la columna Nombre por los valores del 0 hasta el 20 respectivamente
list(nombre.inverse_transform([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]))


#4)	Reemplazamos las categorías de la columna Marca por los valores del 0 al 11 respectivamente
list(marca.inverse_transform([0,1,2,3,4,5,6,7,8,9,10,11]))


#5)	Reemplazamos las categorías de la columna Proveedor por los valores del 0 hasta el 5
list(proveedor.inverse_transform([0,1,2,3,4,5]))

#Podemos observar que ahora las categorías están de forma
#Numérica
#Descomentar la siguiente línea para poder observar el cambio
df.head(10)


#Escogemos la características más importantes del dataframe
caracteristicas = list(df.columns[1:])

#Asignamos los valores de los atributos a las variables
#X y y respectivamente para poder realizar el entrenamiento

#X = df.iloc[:,df.columns != 'Decision'].values
#X = df.iloc[:,df.columns != 'Fecha_venta'].values
X = df[['Nombre','Marca','Proveedor','Precio_venta','Cantidad','Año','Mes','Dia','Total']]
y = df['Decision'].values


#Procedemos a entrenar nuestros algoritmo de clasificador de arbol de decisión
#Ajustando los hiperparámetros de éste

arbol = DecisionTreeClassifier(criterion='gini', max_depth=20, random_state=1)
arbol.fit(X, y)

#obtener las columnas importantes 
importance = arbol.feature_importances_
#sumarizar la importancia de las caracteristicas
for i, v in enumerate(importance):
  if(i<11):
      print('Característica: %0d, Puntaje: %.5f' % (i,v))

#grafica la importancia de cada atributo. Si se ejecuta este archivo .py
#desde la consola, favor cerrar cuando abra ventana del gráfico para 
#que se termine de ejecutar los demás comandos
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


#Procedemos a separar el conjunto de datos de entrenamiento y prueba
X_ent, X_pru, y_ent, y_pru = train_test_split(X, y, random_state = 1)


#Entrenar de nuevo el modelo
# arbol = DecisionTreeClassifier(criterion='entropy', max_depth=30, random_state=1)
# arbol.fit(X, y)


#Procedemos a observar la métrica de exactitud del modelo entrenado
#tanto para los datos del conjunto de entrenamiento, como el 
#del conjunto de prueba 

print("Exactitud del conjunto de entrenamiento: {:.2f}".format(arbol.score(X_ent, y_ent)))
print("Exactitud del conjunto de prueba: {:.2f}".format(arbol.score(X_pru, y_pru)))


#Procedemos a realizar una predicción
#Encargar_Mas
prueba = np.array([13,1,3,167342,892,2021,8,20,149269064])
#Encargar_Menos
#prueba = np.array([19,1,4,48839,156,2021,12,19,7618884])
#Encargar_Normal
#prueba = np.array([8,0,2,6455,448,2022,2,26,2891840])
prediccion = arbol.predict(prueba.reshape(1 , -1))
prediccion


#Ahora, adaptamos la salida numérica de las categorías
#de tal forma que el usuario pueda entenderla
if prediccion[0] == 0:
    print("Encargar_Mas")
elif prediccion[0] == 1:
    print("Encargar_Menos")
elif prediccion[0] == 2:
    print("Encargar_Normal") 


#Como último paso, prodecemos a guardar nuestro modelo
#para un posterior implementación (Interfaz web) y a mostrar
#una gráfica sencilla
x = df['Cantidad']
y = df['Proveedor']


pyplot.bar(x, y)
pyplot.show()



joblib.dump(arbol, "modelo.pkl")
print("El modelo ha sido guardado satisfactoriamente")



















