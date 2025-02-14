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


#2 b)a)

response = requests.get('https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-indices/sunspot-numbers/american/lists/list_aavso-arssn_daily.txt')


if response.status_code == 200:
    text_data = response.text
    df = pd.read_csv(StringIO(text_data), delim_whitespace=True, skiprows=1)
    df_filtered = df[(df["Year"] < 2012) | ((df["Year"] == 2012) & (df["Month"] == 1) & (df["Day"] == 1))]
    df_filtered["date"] = pd.to_datetime(df_filtered[["Year", "Month", "Day"]])
    #s_o = df_filtered["date"][0].timestamp()
    df_filtered["secs"] = (df_filtered["date"].astype('int64') // 10**9) #+ s_o

#print(df_filtered)
freq = np.fft.rfftfreq(len(df_filtered['secs']),abs(df_filtered['secs'][1]-df_filtered['secs'][0]))
FFT = np.fft.rfft(df_filtered['SSN'])
i = np.argmax(abs(FFT)[1:])
#print(i+1)
f = freq[i+1]
T = (1/f) / (60 * 60 * 24 * 365)
print('\n '+ '2.b.a) {P_solar = ' + str(round(T,2)) + ' años}')
plt.scatter(df_filtered['date'],df_filtered['SSN'],label='Datos')
#plt.scatter(freq[:50],abs(FFT)[:50])
#plt.show()


#2 b)b)
f_ny = 1/(2 * 60 * 60 * 24 * 365)
fft_n = []
freqs_n = []
fft_ = abs(FFT).copy()
for i in range(len(freq)):
    if freq[i] % f != 0:
        fft_[i] = 0
    elif freq[i] < f_ny:
        fft_n.append(fft_[i])
        freqs_n.append(freq[i])

#print(len(fft_n))
#print(freqs_n)
"""fre = np.zeros(len(fft_))
fre[np.argmax(abs(FFT)[1:]) + 1] = np.max(abs(FFT)[1:])
y_max = np.fft.irfft(fre)"""
#y_filt = np.fft.irfft(fft_)
#plt.scatter(df_filtered['date'],y_filt)
#plt.scatter(freq[:50],abs(FFT)[:50])
#plt.scatter(freq,fft_)
#plt.scatter(freqs_n,fft_n)
#plt.show()

fecha_inicio = df_filtered['date'][0].date()
fecha_final = date(2025, 2, 10)
#fecha_final = date(2012, 1, 1)
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
        inv[i]=abs(suma)*(1/(N))
    return inv

invers=Inversa(fft_n,freqs_n,len(df_filtered["SSN"].values),secs)
#invers=Inversa(fft_,freq,len(df_filtered["SSN"].values),secs)

print('\n '+ '2.b.b) {n_manchas_hoy = ' + str(int(invers[-1])) + ' }')
plt.plot(rango_fechas,invers,color='r',label='Modelo')
plt.ylabel('# de manchas')
plt.xlabel('Fecha')
plt.legend()
plt.savefig('Tarea 2/2.b.pdf')