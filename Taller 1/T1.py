import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

#print(len(array_inten_new[:255])+len(pol)+len(array_inten_new[410:]), len(espectro_y))

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


"""def FWHM(array_x,array_y):
    x = array_x
    y = array_y
    i = np.argmax(y)
    x_bar = x[i]
    d = x - x_bar
    sigma = np.sqrt(np.sum(d**2)/(len(x)-1))
    return 2*np.sqrt(2*np.log(2))*sigma
    
print(FWHM(pico1_x,pico1),FWHM(pico2_x,pico2))"""

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


print(espectro_max,pico1_max,pico2_max)
print(anchura(espectro_x,espectro_y),anchura(pico1_x,pico1),anchura(pico2_x,pico2))


#plt.plot(array_wave_new, GetModel(array_wave_new,coeficientes[::-1])) #modelo
plt.plot(array_wave_new,array_inten_new) #datos filtraos
plt.plot(espectro_x,espectro_y,color= 'red') #espectro
plt.plot(array_wave_new,array_inten_new - GetModel(array_wave_new,coeficientes[::-1]),color= 'green') #picos

#plt.scatter(pico1_x, pico1,marker='x',color= 'red')
#plt.scatter(pico2_x, pico2,marker='x',color= 'red')

plt.show()