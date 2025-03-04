from IPython.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from mpl_toolkits.mplot3d import Axes3D
from numba import njit

plt.rcParams["animation.html"] = "jshtml"
plt.rcParams["image.origin"] = "upper"
plt.style.use('dark_background')

#Punto 1
N = 300
d = 2 #longitud de la figura
p = 0.0001 #presición
iter_max = 15000

phi = np.random.rand(N,N)
x = np.linspace(-1.1,1.1,N)
y = np.linspace(-1.1,1.1,N)
h = abs(x[0]-x[1]) #separación de nodos

for i in range(N):
    for j in range(N):
        r=np.sqrt(x[i]**2+y[j]**2)
        if r >= 1:
            theta = np.arctan2(y[j], x[i])
            phi[i,j] = np.sin(7*theta)
            
@njit
def iterar_poisson(phif, x, y, n, delta, tol, iter_max):
    for iteration in range(iter_max):
        phi_old = phif.copy()
        for i in range(1, n-1):
            for j in range(1, n-1):
                r = np.sqrt(x[i]**2 + y[j]**2)
                if r < 1: 
                    phif[i, j] = ((phi_old[i+1, j] + phi_old[i-1, j] + phi_old[i, j+1] + phi_old[i, j-1])/4)*(2/3) + ((phi_old[i+1, j+1] + phi_old[i-1, j+1] + phi_old[i+1, j-1] + phi_old[i-1, j-1])/4)*(1/3) +  np.pi * (-x[i] - y[j]) * delta**2
        
        error = np.abs(phif - phi_old).max() 
        if error < tol:
            break
    return phif

phi = iterar_poisson(phi, x, y, N, h, p, iter_max)

X, Y = np.meshgrid(x, y)  
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, phi, cmap='jet')
ax.view_init(elev=30, azim=90)
ax.set_title('Poisson en un Disco')
fig.savefig('Tarea 3 b/1.png')


#Punto 2

# Parámetros de la simulación
longitud = 2       # Longitud de la cuerda
tiempo_total = 1   # Tiempo total de simulación
velocidad = 1      # Velocidad de propagación de la onda
delta_x = 0.01     # Paso espacial
delta_t = 0.005    # Paso temporal

# Número de puntos en el espacio y en el tiempo
num_puntos_x = int(longitud / delta_x)
num_puntos_t = int(tiempo_total / delta_t)

# Coeficiente de Courant (debe ser <= 1 para estabilidad)
coef_courant = velocidad * delta_t / delta_x 

# Discretización del espacio
eje_x = np.arange(0, longitud, delta_x)

# Función para resolver la ecuación de onda con una condición de frontera dada
def resolver_ecuacion_onda(matriz_solucion, condicion_frontera):
    # Condición inicial: una función gaussiana centrada en x = 1/2
    matriz_solucion[0] = np.exp(-125 * (eje_x - 1/2) ** 2)
    matriz_solucion[0] = condicion_frontera(matriz_solucion[0])  # Aplicar condición de frontera
    
    # Primer paso temporal (usando esquema de diferencias finitas)
    for i in range(1, num_puntos_x - 1):
        matriz_solucion[1, i] = matriz_solucion[0, i] + 0.5 * coef_courant**2 * (
            matriz_solucion[0, i+1] - 2 * matriz_solucion[0, i] + matriz_solucion[0, i-1]
        )
    matriz_solucion[1] = condicion_frontera(matriz_solucion[1])  # Aplicar condición de frontera en el siguiente paso
    
    # Iteración en el tiempo
    for n in range(1, num_puntos_t - 1): 
        for i in range(1, num_puntos_x - 1): 
            matriz_solucion[n+1, i] = (
                2 * matriz_solucion[n, i] - matriz_solucion[n-1, i] 
                + coef_courant**2 * (matriz_solucion[n, i+1] - 2 * matriz_solucion[n, i] + matriz_solucion[n, i-1])
            )
        matriz_solucion[n+1] = condicion_frontera(matriz_solucion[n+1])  # Aplicar condición de frontera
    
    return matriz_solucion

# Funciones para aplicar diferentes condiciones de frontera
def frontera_dirichlet(matriz):
    matriz[0] = 0    # Extremo izquierdo fijo en 0
    matriz[-1] = 0   # Extremo derecho fijo en 0
    return matriz

def frontera_neumann(matriz):
    matriz[0] = matriz[1]     # Derivada nula en el extremo izquierdo
    matriz[-1] = matriz[-2]   # Derivada nula en el extremo derecho
    return matriz

def frontera_periodica(matriz):
    matriz[0] = matriz[-2]    # Condición periódica en el extremo izquierdo
    matriz[-1] = matriz[1]    # Condición periódica en el extremo derecho
    return matriz

# Inicializar matrices de la solución con NaN
solucion_dirichlet = np.zeros((num_puntos_t, num_puntos_x)) * np.nan
solucion_neumann = np.zeros((num_puntos_t, num_puntos_x)) * np.nan
solucion_periodica = np.zeros((num_puntos_t, num_puntos_x)) * np.nan

# Resolver la ecuación de onda para cada condición de frontera
solucion_dirichlet = resolver_ecuacion_onda(solucion_dirichlet, frontera_dirichlet)
solucion_neumann = resolver_ecuacion_onda(solucion_neumann, frontera_neumann)
solucion_periodica = resolver_ecuacion_onda(solucion_periodica, frontera_periodica)

# Configurar la figura y los ejes para la animación
figura, ejes = plt.subplots(3, 1, figsize=(6, 8))

# Configurar límites y etiquetas de los ejes
for eje in ejes:
    eje.set_xlim(0, longitud)
    eje.set_ylim(np.min(solucion_dirichlet), np.max(solucion_dirichlet))

# Graficar las curvas de onda con diferentes condiciones de frontera
linea_dirichlet, = ejes[0].plot(eje_x, solucion_dirichlet[0], 'b')  # Azul
linea_neumann, = ejes[1].plot(eje_x, solucion_neumann[0], 'g')      # Verde
linea_periodica, = ejes[2].plot(eje_x, solucion_periodica[0], 'r')   # Rojo

# Etiquetas de cada condición de frontera
ejes[0].text(0.05, 0.9, 'Condición de frontera: Dirichlet', transform=ejes[0].transAxes)
ejes[1].text(0.05, 0.9, 'Condición de frontera: Neumann', transform=ejes[1].transAxes)
ejes[2].text(0.05, 0.9, 'Condición de frontera: Periódica', transform=ejes[2].transAxes)

# Función para actualizar los cuadros de la animación
def actualizar_animacion(frame):
    linea_dirichlet.set_ydata(solucion_dirichlet[frame])
    linea_neumann.set_ydata(solucion_neumann[frame])
    linea_periodica.set_ydata(solucion_periodica[frame])
    return linea_dirichlet, linea_neumann, linea_periodica

# Crear la animación
animacion = animation.FuncAnimation(figura, actualizar_animacion, frames=range(num_puntos_t), interval=50, blit=False)

# Guardar la animación en formato MP4
animacion.save('Tarea 3 b/2.mp4', writer='ffmpeg', fps=30)


#Punto 3

# Parámetros de simulación
total_time = 2
alpha = 0.022
time_step = 0.0001
num_space_points = 200
num_time_steps = int(total_time / time_step)
space_step = 2 / num_space_points

# Espacio en el eje x
x_values = np.linspace(0, 2, num_space_points)

# Inicialización de la matriz de solución
solution_matrix = np.zeros((num_time_steps, num_space_points))

# Condición inicial
solution_matrix[0] = np.cos(np.pi * x_values)

# Primer paso de tiempo
for i in range(num_space_points): 
    solution_matrix[1, i] = solution_matrix[0, i] - 1/3 * time_step/space_step * (solution_matrix[0, np.mod(i+1, num_space_points)] + solution_matrix[0, i] + solution_matrix[0, np.mod(i-1, num_space_points)]) * (solution_matrix[0, np.mod(i+1, num_space_points)] - solution_matrix[0, np.mod(i-1, num_space_points)])- (alpha**2) * (time_step/(space_step)**3) * (solution_matrix[0, np.mod(i+2, num_space_points)] - 2*solution_matrix[0, np.mod(i+1, num_space_points)] + 2*solution_matrix[0, np.mod(i-1, num_space_points)] - solution_matrix[0, np.mod(i-2, num_space_points)])
    

# Función para iterar en el tiempo
def iterate_solution(solution, num_iterations, num_space_points, time_step, space_step, alpha):
    for n in range(1, num_iterations-1): 
        for i in range(num_space_points): 
            solution[n+1, i] = solution[n-1, i] - 1/3 * time_step/space_step * (solution[n, np.mod(i+1, num_space_points)] + solution[n, i] + solution[n, np.mod(i-1, num_space_points)]) * (solution[n, np.mod(i+1, num_space_points)] - solution[n, np.mod(i-1, num_space_points)])- (alpha**2) * (time_step/(space_step)**3) * (solution[n, np.mod(i+2, num_space_points)] - 2*solution[n, np.mod(i+1, num_space_points)] + 2*solution[n, np.mod(i-1, num_space_points)] - solution[n, np.mod(i-2, num_space_points)])
    return solution

# Iterar la solución
solution_matrix = iterate_solution(solution_matrix, num_time_steps, num_space_points, time_step, space_step, alpha)

# Configuración de la animación
fig = plt.figure()
ax = plt.axes(xlim=(0, 2), ylim=(np.min(solution_matrix), np.max(solution_matrix)))
line, = ax.plot(x_values, solution_matrix[0], 'b')  

# Función para actualizar la animación
def update_animation(frame):
    line.set_ydata(solution_matrix[frame]) 
    return line

# Crear la animación
anim = animation.FuncAnimation(fig, update_animation, frames=range(0, len(solution_matrix),200), interval=50, blit=False)
anim.save('Tarea 3 b/3.a.mp4', writer='ffmpeg', fps=30)

# Graficar la solución en 2D
time_values = np.linspace(0, total_time, num_time_steps) 
plt.figure(figsize=(12, 3))
plt.imshow(solution_matrix.T, aspect='auto', cmap='magma', origin='lower',
           extent=[time_values.min(), time_values.max(), x_values.min(), x_values.max()])
plt.xlim(0, 1.5)
plt.colorbar(label= r'$\Psi (x, t)$')
plt.xlabel(r'Tiempo [s]')
plt.ylabel(r'Ángulo x [m]')
plt.savefig('Tarea 3 b/3.a.pdf')


# Calcular las cantidades conservadas
mass = np.sum(solution_matrix, axis=1) * space_step
momentum = np.sum(solution_matrix**2, axis=1) * space_step
energy = np.sum((1/3 * solution_matrix**3 - (alpha * np.gradient(solution_matrix, axis=1))**2), axis=1) * space_step

# Crear la figura con 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

# Graficar la masa
ax1.plot(time_values, mass, label='Masa')
ax1.set_ylabel('Masa')
ax1.legend()
ax1.grid()

# Graficar el momento
ax2.plot(time_values, momentum, label='Momento', color='orange')
ax2.set_ylabel('Momento')
ax2.legend()
ax2.grid()

# Graficar la energía
ax3.plot(time_values, energy, label='Energía', color='green')
ax3.set_ylabel('Energía')
ax3.set_xlabel('Tiempo')
ax3.legend()
ax3.grid()

# Ajustar el layout y guardar la figura
plt.tight_layout()
plt.savefig('Tarea 3 b/3.b.pdf')


#Punto 4

L = 2
Ly = L
Lx = 1
wy = 0.04
wx = 0.4
c = 0.5
c_l = c/5
T = 2
N = 70
n_t = 100
dt = T/n_t
p = 0.01 #presición
iter_max = 15000

x = np.linspace(0,Lx,N+1)
y = np.linspace(0,Ly,2*N+1)
t = np.linspace(0,T,n_t+1)

phi = np.zeros((n_t+1,N+1,2*N+1))
h = abs(x[0]-x[1])
i_x_min = int((0.5-wx/2)/h)
i_x_max = int((0.5+wx/2)/h)
i_y_min = int((1-wy/2)/h)
i_y_max = int((1+wy/2)/h)

def Boundarie_values(phi_t):
    phi_t[:,0] = 0
    phi_t[:,-1] = 0
    phi_t[0,:] = 0
    phi_t[-1,:] = 0
    phi_t[:i_x_min+1,i_y_min:i_y_max+1] = 0
    phi_t[i_x_max:,i_y_min:i_y_max+1] = 0
    return phi_t

#Condiciones iniciales

i_0 = int(0.5/h)
phi[0,i_0,i_0] = 1
phi[0] = Boundarie_values(phi[0])

for i in range(1,N-1):
    for j in range(1,2*N-1):
        phi[1,i,j] = phi[0,i,j] + 0.5 * ((c*dt/h)**2) * (- 4*phi[1, i, j] + (phi[1, i+1, j] + phi[1, i-1, j] + phi[1, i, j+1] + phi[1, i, j-1])*(2/3) + (phi[1, i+1, j+1] + phi[1, i-1, j+1] + phi[1, i+1, j-1] + phi[1, i-1, j-1])*(1/3))
phi[1] = Boundarie_values(phi[1])

@njit
def iterar_wave_function(phi, N, n_t, h, dt, c, tol, iter_max):
    for iteration in range(iter_max):
        phi_old = phi.copy()
        for k in range(1,n_t-1):
            for i in range(1,N-1):
                for j in range(1,2*N-1):
                    if ((x[i]-L/4)**2 + 3*(y[j]-L/2)**2 <= 1/25) and y[i]>=1:
                        phi[k+1,i,j] = 2*phi[k,i,j] - phi[k-1,i,j] + ((c_l*dt/h)**2) * (- 4*phi[k, i, j] + (phi[k, i+1, j] + phi[k, i-1, j] + phi[k, i, j+1] + phi[k, i, j-1])*(2/3) + (phi[k, i+1, j+1] + phi[k, i-1, j+1] + phi[k, i+1, j-1] + phi[k, i-1, j-1])*(1/3))
                    else:
                        phi[k+1,i,j] = 2*phi[k,i,j] - phi[k-1,i,j] + ((c*dt/h)**2) * (- 4*phi[k, i, j] + (phi[k, i+1, j] + phi[k, i-1, j] + phi[k, i, j+1] + phi[k, i, j-1])*(2/3) + (phi[k, i+1, j+1] + phi[k, i-1, j+1] + phi[k, i+1, j-1] + phi[k, i-1, j-1])*(1/3))
            phi[k+1,:,0] = 0
            phi[k+1,:,-1] = 0
            phi[k+1,0,:] = 0
            phi[k+1,-1,:] = 0
            phi[k+1,:i_x_min+1,i_y_min:i_y_max+1] = 0
            phi[k+1,i_x_max:,i_y_min:i_y_max+1] = 0
    return phi

phi = iterar_wave_function(phi,N,n_t,h,dt,c,p,iter_max)

fig, ax = plt.subplots()
cmap = ax.imshow(phi[0], origin='lower', cmap='jet')
fig.colorbar(cmap) 

def frame(i):
    cmap.set_array(phi[i])
    return cmap

anim4 = animation.FuncAnimation(fig, frame, frames=range(1, len(t)-1))
anim4.save("Tarea 3 b/4.a.mp4",writer='ffmpeg',fps=30)