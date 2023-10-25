import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier


#Titulo
st.header("Proyecto final Electiva cc")




#Atributos
#'Daytime/evening attendance\t','Nacionality',
#'Admission grade','Educational special needs','International'

#Formulario
#Entrada 1 daytime<


pieza = st.selectbox("Escoje una pieza de la moto",
('Cilindros', 'Bombillo trasero', 'Bielas', 'Espejos retrovisores', 
'Escape', 'Acelerador', 'Maneta derecha', 'Maneta izquiera', 'Chasis',
'Pistón del motor', 'Carburador', 'Embrague', 'Rines', 'Neumáticos',
'Amortiguador', 'Bombillo delantero', 'Árbol de levas', 'Cadena',
'Reposapie', 'Depósito de gasolina', 'Claxon'))

if (pieza == 'Cilindros'):
    pieza = 0
if (pieza == 'Bombillo trasero'):
    pieza = 1
if (pieza == 'Bielas'):
    pieza = 2
if (pieza == 'Espejos retrovisores'):
    pieza = 3
if (pieza == 'Escape'):
    pieza = 4
if (pieza == 'Acelerador'):
    pieza = 5
if (pieza == 'Maneta derecha'):
    pieza = 6
if (pieza == 'Maneta izquierda'):
    pieza = 7
if (pieza == 'Chasis'):
    pieza = 8
if (pieza == 'Pistón del motor'):
    pieza = 9
if (pieza == 'Carburador'):
    pieza = 10
if (pieza == 'Embrague'):
    pieza = 11
if (pieza == 'Rines'):
    pieza = 12
if (pieza == 'Neumáticos'):
    pieza = 13
if (pieza == 'Amortiguador'):
    pieza = 14
if (pieza == 'Bombillo delantero'):
    pieza = 15
if (pieza == 'Árbol de levas'):
    pieza = 16
if (pieza == 'Cadena'):
    pieza = 17
if (pieza == 'Reposapie'):
    pieza = 18
if (pieza == 'Depósito de gasolina'):
    pieza = 19
if (pieza == 'Claxon'):
    pieza = 20



marca = st.selectbox("Escoje una marca de la pieza",
('Akt','kymco','Suzuki','YCF Riding','BMW','Apollo motor',
'Pulsar','Ducati','Cf Moto','Yamaha','Aprilia','Honda'))

if (marca == 'Akt'):
    marca = 0
if (marca == 'kymco'):
    marca = 1
if (marca == 'Suzuki'):
    marca = 2
if (marca == 'YCF Riding'):
    marca = 3
if (marca == 'BMW'):
    marca = 4
if (marca == 'Apollo motor'):
    marca = 5
if (marca == 'Pulsar'):
    marca = 6
if (marca == 'Ducati'):
    marca = 7
if (marca == 'Cf Moto'):
    marca = 8
if (marca == 'Yamaha'):
    marca = 9
if (marca == 'Aprilia'):
    marca = 10
if (marca == 'Honda'):
    marca = 11


proveedor = st.selectbox("Escoje un proveedor de la pieza",
('Chaoyang','KIXX','Bajaj','GC Motor','Yahosuka','Everest'))

if (proveedor == 'Chaoyang'):
    proveedor = 0
if (proveedor == 'KIXX'):
    proveedor = 1
if (proveedor == 'Bajaj'):
    proveedor = 2
if (proveedor == 'GC Motor'):
    proveedor = 3
if (proveedor == 'Yahosuka'):
    proveedor = 4
if (proveedor == 'Everest'):
    proveedor = 5


precio_venta = st.number_input("Digite el precio de la pieza")

cantidad = st.number_input("Ingrese la cantidad de piezas a vender")

año = st.number_input("Digite el año de la venta")

mes = st.number_input("Digite el mes de la venta")

dia = st.number_input("Digite el dia de la ventaa")

total = cantidad*precio_venta


#si el boton predecir es presionado

if st.button("Submit"):

    #Cargamos nuestro modelo
    modelo = joblib.load("modelo.pkl")

    #Guardamos las entradas dentro del dataframe

    prediccion = modelo.predict([[pieza, marca, proveedor, precio_venta, cantidad, año, mes, dia, total]])

    #Ahora, adaptamos la salida numérica de las categorías
    #de tal forma que el usuario pueda entenderla
    if prediccion[0] == 0:
        resultado = "Encargar Mas Piezas al proveedor"

    elif prediccion[0] == 1:
        resultado = "Encargar Menos Piezas al proveedor"
       
    elif prediccion[0] == 2:
        resultado = "Encargar Normalmente al proveedor"
       

    #Mensaje de la repuesta de la predicción

    st.text(f"Usted debe : {resultado}")
    #st.text(prediccion)
