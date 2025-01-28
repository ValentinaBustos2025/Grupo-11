import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import trapezoid
import re


def prom(array):
    return np.sum(array) / np.size(array)

df1 = pd.read_csv("Taller 1\Rhodium.csv")

array_wave = df1['Wavelength (pm)'].to_numpy(); array_inten = df1['Intensity (mJy)'].to_numpy()
array_wave = np.append(array_wave, array_wave[-1]); array_inten = np.append(array_inten, array_inten[-1])


def metodo_alvaro(array_x, array_y):
    array_x_new = [array_wave[0]]; array_y_new = [array_inten[0]]
    contador = 1
    if  np.size(array_x)%2 == 0:
        arrays = np.split(array_y, 2)

        for array in arrays:
            prome = prom(array)
            for i in range(0, np.size(array)-1):
                if abs(array[i+1]-array[i]) <= abs(prome*0.3):
                    array_x_new.append(array_x[contador])
                    array_y_new.append(array_y[contador])
                contador += 1
        return np.array(array_x_new),np.array(array_y_new)
    
    else:
        return


def metodo_sergi(array_x, array_y,w):
    array_x_new = [array_wave[0]]; array_y_new = [array_inten[0]]
    prome = prom(array_y)
    for i in range(0,len(array_y)-1):
        if (array_y[i+1]-array_y[i]) <= w*prome:
            array_x_new.append(array_x[i+1])
            array_y_new.append(array_y[i+1])
    return np.array(array_x_new),np.array(array_y_new)


def GetModel(x,p):
    
    y = 0.
    for i in range(len(p)):
        y += p[i]*x**i
        
    return y
 

array_wave_new, array_inten_new = metodo_sergi(array_wave, array_inten,0.3)


array_wave_1 = np.split(array_wave_new, 2)[0]
array_inten_1 = np.split(array_inten_new, 2)[0]

array_wave_new, array_inten_new = metodo_alvaro(array_wave, array_inten) 

array_wave_2 = np.split(array_wave_new, 2)[1]
array_inten_2 = np.split(array_inten_new, 2)[1]


array_wave_new = np.concatenate((array_wave_1,array_wave_2))
array_inten_new = np.concatenate((array_inten_1,array_inten_2))

array_wave_new = np.delete(array_wave_new,-1)
array_inten_new = np.delete(array_inten_new,-1)

w = np.delete(array_wave_new,slice(255,410))
I = np.delete(array_inten_new,slice(255,410))


coeficientes, covarianza = np.polyfit(w, I, 16, cov=True)
pol = GetModel(array_wave_new[255:410],coeficientes[::-1])

espectro_x = array_wave_new
espectro_y = np.concatenate((np.concatenate((array_inten_new[:255],pol)), array_inten_new[410:]))

pico1_max = np.max(array_inten_new[255:410] - pol)

espectro_max = np.max(I)

i = np.argmax(array_inten_new[255:410] - pol)
n = 21
pico1 = (array_inten_new[255:410] - pol)[i-n:i+n]
pico1_x = (array_wave_new[255:410])[i-n:i+n]

pico2_max = np.max((array_inten_new[255:410] - pol)[i+n:])
j = np.argmax((array_inten_new[255:410] - pol)[i+n:])
m = 25
pico2 = ((array_inten_new[255:410] - pol)[i+n:])[j-m:j+m]
pico2_x = ((array_wave_new[255:410])[i+n:])[j-m:j+m]


def anchura(array_x,array_y):
    x = array_x
    y = array_y
    max = np.max(y)
    pos = np.argmax(y)
    for i in range(pos):
        if y[i] <= max/2:
            x1 = x[i]
    for i in range(len(x)-pos):
        if y[-i] <= max/2:
            x2 = x[-i]
    return x2-x1


energia = trapezoid(array_inten_new, array_wave_new)
energia = format(energia, ".2f")
delta_energia = (array_inten_new * 0.02) ** 2
delta_energia_integral = np.sqrt(np.sum(delta_energia))



print("1.a)  Numero de datos eliminados: " + str(1199 - np.size(array_wave_new))) 


print("1.b)  Método: Restar modelo espectro de fondo")



print("1.c ) máximo del espectro es: " + str(espectro_max) + ", el máximo del pico 1 es: " + str(pico1_max) + " yel máximo del pico 2 es: " + str(pico2_max))
print("La anchura de el espectro de fondo es: " + str(round(anchura(espectro_x,espectro_y),2)) + ", la anchura del pico 1 es: " + str(anchura(pico1_x,pico1)) + " y la anchura del pico 2 es: " + str(anchura(pico2_x,pico2)))



print(f"1.d)  La energia total es: {energia}"  + "mJ  y la incertidumbre es: +/- " + str(round(delta_energia_integral, 2)) + "mJ")


plt.figure(figsize=(10,6))
plt.scatter(array_wave,array_inten)
plt.scatter(array_wave_new,array_inten_new,marker='x',color='red') 
plt.title("Limpieza")
plt.xlabel("Wavelength (pm)")
plt.ylabel("Intensity (mJy)")
plt.grid(True)
plt.savefig("Taller 1\limpieza.pdf")
plt.show() 

#plt.plot(array_wave_new, GetModel(array_wave_new,coeficientes[::-1])) #modelo
#plt.plot(array_wave_new,array_inten_new) #datos filtraos
#plt.plot(espectro_x,espectro_y,color= 'red') #espectro

plt.figure(figsize=(10,6))
plt.plot(array_wave_new,array_inten_new - GetModel(array_wave_new,coeficientes[::-1]),color= 'green') #picos
plt.title("Picos de Rayos x")
plt.xlabel("Wavelength (pm)")
plt.ylabel("Intensity (mJy)")
plt.grid(True)
plt.savefig("Taller 1\picos.pdf")
plt.show() 

#plt.scatter(pico1_x, pico1,marker='x',color= 'red')
#plt.scatter(pico2_x, pico2,marker='x',color= 'red')


def procesar_linea(linea):
    
    linea_corregida = re.sub(r'(\d\.\d+)(?=\d)', r'\1 ', linea)
    numeros = re.findall(r'-?\d+\.\d+', linea_corregida)
    
    return [float(num) for num in numeros]


archivo = 'Taller 1/hysteresis.dat'
with open(archivo, 'r') as file:
    lineas = file.readlines()


datos_procesados = []
lineas_problematicas = []

for i, linea in enumerate(lineas):
    try:
        valores = procesar_linea(linea)
        if len(valores) == 3:  
            datos_procesados.append(valores)
        else:
            lineas_problematicas.append((i + 1, linea.strip())) 
    except Exception as e:
        lineas_problematicas.append((i + 1, linea.strip()))


df = pd.DataFrame(datos_procesados)
df.dropna()
df = df.apply(pd.to_numeric, errors='coerce')


fig, ax = plt.subplots(2,1, figsize=(10,6), layout='constrained')
ax[0].scatter(df[0],df[1])
ax[0].set_title("Tiempo (ms) vs Campo Magnético (mT)")
ax[0].set_xlabel("Tiempo (ms)")
ax[0].set_ylabel("B (mT)")
ax[0].grid(True)

ax[1].scatter(df[0],df[2])
ax[1].set_title("Tiempo (ms) vs Campo Interno (A/m)")
ax[0].set_xlabel("Tiempo (ms)")
ax[1].set_ylabel("H (A/m)")
ax[1].grid(True)
plt.savefig("Taller 1\histérico.pdf")
plt.show() 


#Punto 2.2
for i in range(0,len(df)):
    if df[1][i] == df[1][0] : 
        T = round(1/((df[0][i]*2)*10e-3),2)

print("El periodo es {0} Hz".format(T))


#Punto 2.3
plt.figure(figsize=(10,6))
plt.scatter(df[1], df[2])
plt.title("Histerésis Magnética")
plt.xlabel("H (A/m)")
plt.ylabel("B (mT)")
plt.grid(True)
plt.savefig("Taller 1\energy.pdf")
plt.show() 

area_hysteresis = np.trapz(df[2], df[1])
print(area_hysteresis)