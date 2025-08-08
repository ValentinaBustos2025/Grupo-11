import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import cmath
from scipy.optimize import curve_fit
from numba import njit, complex128
import requests
import pandas as pd
from io import StringIO
from datetime import date
import re

import warnings
warnings.filterwarnings("ignore")

def datos_prueba(t_max:float, dt:float, amplitudes:NDArray[float], 
                 frecuencias:NDArray[float], ruido:float=0.0) -> NDArray[float]:
    ts = np.arange(0.,t_max,dt)
    ys = np.zeros_like(ts,dtype=float)
    for A,f in zip(amplitudes,frecuencias):
        ys += A*np.sin(2*np.pi*f*ts)
    ys += np.random.normal(loc=0,size=len(ys),scale=ruido) if ruido else 0
    return ts,ys

# 1.a)
@njit
def Fourier(t: np.ndarray, y: np.ndarray, f: np.ndarray) -> np.ndarray:
    r = np.zeros(len(f), dtype=np.complex128)
    for j in range(len(f)):
        r[j] = np.sum(y * np.exp(-2j * np.pi * t * f[j]))
    return r

def plot_fourier(frecuencias: np.ndarray, fourier_con_ruido: np.ndarray, fourier_sin_ruido: np.ndarray) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 6), layout='constrained')
    ax[0].plot(frecuencias, np.abs(fourier_sin_ruido))
    ax[0].set_title("Frecuencia (Hz) vs Magnitud sin ruido")
    ax[0].set_xlabel("Frecuencia (Hz)")
    ax[0].set_ylabel("Magnitud")
    ax[0].grid(True)
    ax[1].plot(frecuencias, np.abs(fourier_con_ruido))
    ax[1].set_title("Frecuencia (Hz) vs Magnitud con ruido")
    ax[1].set_xlabel("Frecuencia (Hz)")
    ax[1].set_ylabel("Magnitud")
    ax[1].grid(True)
    plt.savefig("Tarea 2/1.a.pdf")


t_max = 1.0; dt = 0.001; ruido = 0.01 ; amplitudes = np.array([1.0, 1.5, 1.2]); frecuencias = np.array([500.0, 50.0, 200.0])
ts, ys_sin_ruido = datos_prueba(t_max, dt, amplitudes, frecuencias, ruido=0.0)
_, ys_con_ruido = datos_prueba(t_max, dt, amplitudes, frecuencias, ruido=ruido)
frecuencias = np.linspace(0, 300, 50)
Fourier_sin_ruido = Fourier(ts, ys_sin_ruido, frecuencias)
Fourier_con_ruido = Fourier(ts, ys_con_ruido, frecuencias)


plot_fourier(frecuencias, Fourier_con_ruido, Fourier_sin_ruido)
print("1.a) Aumento de ruido magnifica todas las frecuencias y no se encuentran los picos principales.")

# 1.b)
################################################################################################
@njit
def encontrar_FWHM(magnitudes: np.ndarray, frecuencias: np.ndarray) -> float:
    indice_pico = np.argmax(magnitudes)
    altura_pico = magnitudes[indice_pico]
    altura_media = altura_pico / 2
    cruces = np.where(magnitudes >= altura_media)[0]
    izquierda = cruces[0]
    derecha = cruces[-1]
    return frecuencias[derecha] - frecuencias[izquierda]

def FWHM_diferentes_t(amplitudes: np.ndarray, frecuencias: np.ndarray, t_max: np.ndarray, dt: float = 0.001) -> np.ndarray:
    anchuras = np.zeros(len(t_max))
    for i, t in enumerate(t_max):
        ts_nuevo, y_nuevo_sin_ruido = datos_prueba(t, dt, amplitudes, frecuencias, ruido=0.0)
        freq = np.linspace(0, 300, 500)
        Fourier_sin_ruido_nuevo = np.abs(Fourier(ts_nuevo, y_nuevo_sin_ruido, freq))
        fwhm = encontrar_FWHM(Fourier_sin_ruido_nuevo, freq)
        anchuras[i] = fwhm
    return anchuras

def plot_fwhm(t_max_values: np.ndarray, fwhm_values: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    ax.loglog(t_max_values, fwhm_values, 'o-', label='FWHM')
    ax.set_title("FWHM para diferentes tiempos")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("FWHM (Hz)")
    ax.grid(True)
    plt.savefig("Tarea 2/1.b.pdf")

# Generación de datos
amplitudes = np.array([1.0]); frecuencias = np.array([50.0]); t_max_values = np.linspace(10, 300, 500)
fwhm_values = FWHM_diferentes_t(amplitudes, frecuencias, t_max_values)
plot_fwhm(t_max_values, fwhm_values)


#1.c)

#Me robé lo de la tarea pasada c: 
################################################################################################

def procesar_linea(linea):
    
    linea_corregida = re.sub(r'(\d\.\d+)(?=\d)', r'\1 ', linea)
    numeros = re.findall(r'-?\d+\.\d+', linea_corregida)
    
    return [float(num) for num in numeros]


archivo = 'Tarea 2/OGLE-LMC-CEP-0001.dat'
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

df = pd.DataFrame(datos_procesados) ; df.dropna() ;  df = df.apply(pd.to_numeric, errors='coerce')

def calcular_frecuencia_nyquist(tiempo):
    dt = np.mean(np.diff(tiempo))
    frecuencia_muestreo = 1.0 / dt
    frecuencia_nyquist = frecuencia_muestreo / 2
    return frecuencia_nyquist

def encontrar_frecuencia_dominante(tiempo, intensidad):
    dt = np.mean(np.diff(tiempo))
    frecuencias = np.fft.fftfreq(len(tiempo), dt)
    fft_values = np.fft.fft(intensidad)
    frecuencia_dominante = frecuencias[np.argmax(np.abs(fft_values))]
    return abs(frecuencia_dominante)

def graficar_fase_vs_intensidad(tiempo, intensidad, f_true):
    fase = np.mod(f_true * tiempo, 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(fase, intensidad, s=1, color='blue', label='Datos')
    ax.set_xlabel('Fase (φ)', fontsize=12)
    ax.set_ylabel('Intensidad (y)', fontsize=12)
    ax.set_title('Intensidad en función de la fase', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    plt.savefig('Tarea 2/1.c.pdf', bbox_inches='tight')
    #plt.show()

############################################################################################################

tiempo = np.array(df[0]) ; intensidad = np.array(df[1]) ;  incertidumbre = np.array(df[2]); frequencies = np.linspace(0, 8, len(tiempo))

f_nyquist = calcular_frecuencia_nyquist(tiempo)
print(f"1.c f Nyquist: {f_nyquist}")
f_true = encontrar_frecuencia_dominante(tiempo, intensidad)
print(f"1.c) f true: {f_true}")
graficar_fase_vs_intensidad(tiempo, intensidad, f_true)


############################################################################################################
#2 a)

@njit
def Fourier(t: np.ndarray, y: np.ndarray, f: np.ndarray) -> np.ndarray:
    r = np.zeros(len(f), dtype=np.complex128)
    for j in range(len(f)):
        r[j] = np.sum(y * np.exp(-2j * np.pi * t * f[j]))
    return r

data =pd.read_csv('Tarea 2/H_field.csv').to_numpy().T
t = data[0]
H = data[1]
freq = np.fft.rfftfreq(len(t),abs(t[1]-t[0]))

#numpy
FFT = np.fft.rfft(H)
i = np.argmax(np.abs(FFT)[1:])
f_fast = freq[i+1]

fourier = np.abs(Fourier(t,H,freq))
i = np.argmax(fourier[1:])
f_general = float(np.round(freq[i+1],5))

print(f"2.a) {f_fast = :.5f}; {f_general = }")
f_general = freq[i+1]

def fase(f,t):
    return np.mod(f * t, 1)

fase_fast = fase(f_fast,t)
fase_general = fase(f_general,t)

plt.figure(figsize=(10, 4))
plt.scatter(fase_fast,H,label='f_fast')
plt.scatter(fase_general,H,label='f_general')
plt.ylabel('H')
plt.xlabel('Fase')
plt.legend()
plt.savefig('Tarea 2/2.a.pdf')


#2 b)a)

response = requests.get('https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-indices/sunspot-numbers/american/lists/list_aavso-arssn_daily.txt')


if response.status_code == 200:
    text_data = response.text
    df = pd.read_csv(StringIO(text_data), delim_whitespace=True, skiprows=1)
    df_filtered = df[(df["Year"] < 2012) | ((df["Year"] == 2012) & (df["Month"] == 1) & (df["Day"] == 1))]
    df_filtered["date"] = pd.to_datetime(df_filtered[["Year", "Month", "Day"]])
    df_filtered["secs"] = (df_filtered["date"].astype('int64') // 10**9)


freq = np.fft.rfftfreq(len(df_filtered['secs']),abs(df_filtered['secs'][1]-df_filtered['secs'][0]))
FFT = np.fft.rfft(df_filtered['SSN'])
i = np.argmax(abs(FFT)[1:])
f = freq[i+1]
T = (1/f) / (60 * 60 * 24 * 365)
print('2.b.a) {P_solar = ' + str(round(T,2)) + ' años}')
plt.figure(figsize=(10, 4)) 
plt.scatter(df_filtered['date'],df_filtered['SSN'],label='Datos')


#2 b)b)

fft_n = FFT[:50]
freqs_n = freq[:50] 

fecha_inicio = df_filtered['date'][0].date()
fecha_final = date(2025, 2, 17)
dias_total = (fecha_final - fecha_inicio).days
rango_fechas = pd.date_range(start=fecha_inicio, periods=dias_total + 1, freq='D')
secs = np.arange(0, dias_total + 1) * 60 * 60 * 24 

def Inversa(T_func,f_func,N,t):
    s=len(t)
    m=len(f_func)
    inv=np.zeros(s)
    for i in range(s):
        tj=t[i]
        suma=0
        for k in range(m):
            suma+= (T_func[k]*0.5) * np.exp(2*np.pi*1j*tj*f_func[k])
        inv[i]=np.real(suma)*(1/(N))
    return inv

invers=Inversa(fft_n,freqs_n,len(df_filtered["SSN"].values),secs)

print('2.b.b) {n_manchas_hoy = ' + str(int(invers[-1])) + ' }')
plt.plot(rango_fechas,invers,label='Modelo',color='r')
plt.ylabel('# de manchas')
plt.xlabel('Fecha')
plt.legend()
plt.savefig('Tarea 2/2.b.pdf')


#Punto 3


from PIL import Image
plt.rcParams["figure.figsize"] = (10,3)


archivo = "Tarea 2/list_aavso-arssn_daily.txt"

df = pd.read_csv(archivo, delim_whitespace=True, on_bad_lines='skip', skiprows=2)
df = df[df.notna().sum(axis=1) == 4] 
datos = df.astype(float).to_numpy()

años = datos[:, 0]
meses = datos[:, 1]
dias = datos[:, 2]
SSN = datos[:, 3]

fechas = años + meses / 12 + dias / 365
manchas =  SSN

freq = np.fft.rfftfreq(len(fechas), fechas[1] - fechas[0])
FFT = np.fft.rfft(manchas)

alphas = [0, 1, 3, 10]
fig, axes = plt.subplots(len(alphas), 1, figsize=(10, 4))


for i in range(0, len(alphas)):
    FFT_filtrada = FFT * np.exp(-(freq*alphas[i])**2)
    y_filt = np.fft.irfft(FFT_filtrada)
    axes[i].plot(fechas, manchas, color=str(0.8))
    axes[i].plot(fechas, y_filt, color=str(0.3))
    axes[i].set_title('Alpha = ' + str(alphas[i]))

plt.tight_layout()
plt.savefig("Tarea 2/3.1.pdf", format="pdf", dpi=300, bbox_inches="tight") #GUARDA PDF


import cv2

image = cv2.imread(r'Tarea 2\Noisy_Smithsonian_Castle.jpg', 0)

f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)

fshift[215:550, 408:415] = 0
fshift[215:550, 608:616] = 0
fshift[0:374, 506:518] = 0
fshift[387:765, 506:518] = 0


f_ishift = np.fft.ifftshift(fshift)
img_filtered = np.fft.ifft2(f_ishift)
img_filtered = np.abs(img_filtered)

plt.figure(figsize=(10, 8))
plt.imshow(img_filtered, 
           cmap='gray', 
           vmin=0, 
           vmax=255)
plt.axis('off') 
plt.tight_layout()
plt.savefig('Tarea 2/3.b.a.png')
#plt.show()


image2 = cv2.imread(r'Tarea 2\catto.png', 0)

f = np.fft.fft2(image2)
fshift = np.fft.fftshift(f)
plt.matshow(abs(fshift),cmap="grey",norm="log")

for n in range(210, 370):
    for j in range(375,420):
        fshift[n,j] = 0

for n in range(385, 540):
    for j in range(270,540):
        fshift[n,j] = 0


f_ishift = np.fft.ifftshift(fshift)
img_filtered = np.fft.ifft2(f_ishift)
img_filtered = np.abs(img_filtered)

plt.figure(figsize=(10, 8))
plt.imshow(img_filtered, 
           cmap='gray', 
           vmin=0, 
           vmax=255)
plt.axis('off') 
plt.tight_layout()
plt.savefig("Tarea 2/3.b.b.png", format="png", dpi=300, bbox_inches="tight") #GUARDA PNG#

#ajsjasa