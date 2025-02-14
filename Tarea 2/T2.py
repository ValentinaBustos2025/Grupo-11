import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import cmath
from scipy.optimize import curve_fit
from numba import njit, complex128

def datos_prueba(t_max:float, dt:float, amplitudes:NDArray[float], 
                 frecuencias:NDArray[float], ruido:float=0.0) -> NDArray[float]:
    ts = np.arange(0.,t_max,dt)
    ys = np.zeros_like(ts,dtype=float)
    for A,f in zip(amplitudes,frecuencias):
        ys += A*np.sin(2*np.pi*f*ts)
    ys += np.random.normal(loc=0,size=len(ys),scale=ruido) if ruido else 0
    return ts,ys

#1. a) 

@njit
def Fourier(t:NDArray[float], y:NDArray[float], f:NDArray[float]) -> complex:  
      r = np.zeros((len(f), len(t)), dtype= complex128)
      for j in range(len(f)):
            for i in range(len(t)): 
                r [j,i] = (y[i]* np.exp(-2j * np.pi * t[i] * f[j]))
      return np.sum((r), axis=1 )


def plot_fourier(frecuencias:NDArray[float], fourier_con_ruido: NDArray[float], fourier_sin_ruido : NDArray[float] ) -> None:
        fig, ax = plt.subplots(1,2, figsize=(10,6), layout='constrained')
        ax[0].bar(frecuencias , abs(fourier_sin_ruido),  width = 5  )
        ax[0].set_title("Frecuencia (Hz) vs Magnitud sin ruido ")
        ax[0].set_xlabel("Frecuencia (Hz)")
        ax[0].set_ylabel("Magnitud")
        ax[0].grid(True)
        ax[1].bar(frecuencias , abs(fourier_con_ruido),  width = 5)
        ax[1].set_title("Frecuencia (Hz) vs Magnitud con ruido ")
        ax[1].set_xlabel("Frecuencia (Hz)")
        ax[1].set_ylabel("Magnitud")
        ax[1].grid(True)
        plt.savefig("Tarea 2/1.a.pdf")

###################################################################################################

#Información de Prueba 
t_max = 1.0 ; dt = 0.001 ; ruido = 1 
amplitudes = np.array([1.0, 1.5, 1.2]) ;  frecuencias = np.array([500.0, 50.0, 200.0]) 

# Generar datos 
ts, ys_sin_ruido = datos_prueba(t_max, dt, amplitudes, frecuencias, ruido=0.0)
_, ys_con_ruido = datos_prueba(t_max, dt, amplitudes, frecuencias, ruido=ruido)

#Generación de la Fourier de los datos anteriores
frecuencias= np.linspace(0, 300, 50)
Fourier_sin_ruido= Fourier(ts, ys_sin_ruido, frecuencias )
Fourier_con_ruido = Fourier(ts, ys_con_ruido, frecuencias)

############################################################################################################

plot_fourier(frecuencias, Fourier_con_ruido, Fourier_sin_ruido)
print("1.a) Aumento de ruido magnifica todas las frecuencias y no se encuentran los picos principales.")


#1. b) 


def gaussian(x, a, b, c):
    return a * np.exp(-0.5 * ((x - b) / c) ** 2)


def encontrar_FWHM(a: NDArray[float], f: NDArray[float]) -> float:  
    m = np.argmax(a) ; corte_a = f[max(m - 5, 0):min(m + 5, len(a) - 1)+1]; corte_b = a[ max(m - 5, 0):min(m + 5, len(a) - 1)+1]
    p0 = [a[m], f[m], (f[min(m + 5, len(a) - 1)] - f[ max(m - 5, 0)]) / 2] #Amplitud, Posició de la frecuencia, Ancho Inicial
    popt, pcov = curve_fit(gaussian, corte_a, corte_b, p0=p0)
    c = popt[2]
    fwhm = 2 * np.sqrt(2 * np.log(2)) * c

    return round(fwhm,2) , round(f[m],2), round(a[m] / 2 ,2)   

def FWHM_diferentes_t(amplitudes:NDArray[float] , frecuencias: NDArray[float], t_max:NDArray[float], dt=0.001)->  NDArray[float]:
    generada = []
    for t in t_max:
        ts_nuevo, y_nuevo_sin_ruido = datos_prueba(t, dt, amplitudes, frecuencias, ruido=0.0)
        generada.append((ts_nuevo, y_nuevo_sin_ruido))  


    anchuras = []
    for ts_nuevo, y_nuevo_sin_ruido in generada:
        freq = np.linspace(0, 300, 500)
        Fourier_sin_ruido_nuevo = abs(Fourier(ts_nuevo, y_nuevo_sin_ruido, freq))
        fwhm, _, _ = encontrar_FWHM(Fourier_sin_ruido_nuevo, freq)
        anchuras.append([fwhm])

    return anchuras

def plot_FWHM(anchuras:NDArray[float], t_max:NDArray[float])->None: 
    fig, ax = plt.subplots(1,1, figsize=(10,6), layout='constrained')
    ax.plot(t_max,anchuras)
    plt.show()
    return None

###################################################################################################

amplitudes = np.array([1.0, 1.5, 1.2]) ;  frecuencias = np.array([500.0, 50.0, 200.0]); t_max = np.linspace(0, 100, 10)
anchuras = FWHM_diferentes_t(amplitudes, frecuencias, t_max)
plot_FWHM(anchuras, t_max)
###################################################################################################