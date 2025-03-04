import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation

plt.rcParams["animation.html"] = "jshtml"

#Punto 1

#1.a)
#Condiciones iniciales 
m = 10 ; v0 = 10; g = 9.773 ; b = 0
 
# Ecuaciones de movimiento con fricción
def motion_equations(t, y):
    ux, uy, x, y_pos = y
    speed = np.sqrt(ux**2 + uy**2)**2 
    dux_dt = -b * ux * speed / m
    duy_dt = -g - b * uy * speed / m
    dx_dt = ux ; dy_dt = uy
    return [dux_dt, duy_dt, dx_dt, dy_dt]

#Arreglo para el valor dónde el piso toca el suelo 
def hit_ground(t, y):
    return y[3] 
hit_ground.terminal = True #Termina la integración cuando el evento que tocó el piso acabó
hit_ground.direction = -1 # Cambia la dirección del evento 

angles = np.linspace(0, 90, 91); ranges = []; max_range = 0; best_angle = 0

for angle in angles:
    rad = np.radians(angle)
    initial_conditions = [v0 * np.cos(rad), v0 * np.sin(rad), 0, 0]
    sol = solve_ivp(motion_equations, [0, 10], initial_conditions, events=hit_ground, dense_output=False)
    
    if sol.t_events[0].size > 0: 
        range_at_angle = sol.y_events[0][0, 2]
        ranges.append(range_at_angle)
        if range_at_angle > max_range:
            max_range = range_at_angle
            best_angle = angle
    else:
        ranges.append(0)

print(f"El ángulo óptimo para el coeficiente de 𝛽 {b} para el máximo alcance es {best_angle}° con un alcance de {max_range:.2f} metros.")

#1.b) 
#Energía disipada
E_initial = 0.5 * m * v0**2 ; betas = np.linspace(0, 2, 91)

def hit_ground(t, y, beta):
    return y[3]
hit_ground.terminal = True
hit_ground.direction = -1

def motion(t, y, beta):
    ux, uy, x, y_pos, energy_acc = y 
    speed = np.sqrt(ux**2 + uy**2)
    dux_dt = -beta * ux * speed / m
    duy_dt = -g - beta * uy * speed / m
    dx_dt = ux ; dy_dt = uy
    power_friction = beta * speed**3 
    return [dux_dt, duy_dt, dx_dt, dy_dt, power_friction]

best_angles = []; energy_losses = []

for beta in betas:
    max_range = 0; best_angle = 0
    for angle in np.linspace(0, 90, 91):
        rad = np.radians(angle)
        initial_conditions = [v0 * np.cos(rad), v0 * np.sin(rad), 0, 0, 0]
        sol = solve_ivp(motion, [0, 10], initial_conditions, args=(beta,), events=hit_ground, dense_output=False)
        
        if sol.t_events[0].size > 0:
            range_at_angle = sol.y_events[0][0, 2]
            if range_at_angle > max_range:
                max_range = range_at_angle ; best_angle = angle
    
    rad = np.radians(best_angle)
    initial_conditions = [v0 * np.cos(rad), v0 * np.sin(rad), 0, 0, 0]
    sol = solve_ivp(motion, [0, 10], initial_conditions, args=(beta,), events=hit_ground, vectorized=True, dense_output=True)
    if sol.y.shape[1] > 0:
        energy_lost = np.trapz(sol.y[4], sol.t)  
        energy_losses.append(energy_lost)
    else:
        energy_losses.append(0)
    
    best_angles.append(best_angle)


fig, ax = plt.subplots(2, 1, figsize=(10, 10))

ax[0].step(betas, best_angles, where='mid', label='Ángulo óptimo')
ax[0].set_xscale('log')
ax[0].set_title('Ángulo óptimo de lanzamiento vs. Coeficiente de fricción β')
ax[0].set_xlabel('Coeficiente de fricción β')
ax[0].set_ylabel('Ángulo óptimo θ (grados)')
ax[0].grid(True)

ax[1].step(betas, best_angles, where='mid', label='Ángulo óptimo')
ax[1].set_title('Ángulo óptimo de lanzamiento vs. Coeficiente de fricción β')
ax[1].set_xlabel('Coeficiente de fricción β')
ax[1].set_ylabel('Ángulo óptimo θ (grados)')
ax[1].grid(True)
plt.savefig("Tarea 3/1.a.pdf")


fig, ax = plt.subplots(2, 1, figsize=(10, 10))

ax[0].step(betas, energy_losses, where='mid', label='Ángulo óptimo')
ax[0].set_xscale('log')
ax[0].set_title('Ángulo óptimo de lanzamiento vs. Coeficiente de fricción β')
ax[0].set_xlabel('Coeficiente de fricción β')
ax[0].set_ylabel('Ángulo óptimo θ (grados)')
ax[0].grid(True)

ax[1].step(betas, energy_losses, where='mid', label='Ángulo óptimo')
ax[1].set_title('Ángulo óptimo de lanzamiento vs. Coeficiente de fricción β')
ax[1].set_xlabel('Coeficiente de fricción β')
ax[1].set_ylabel('Ángulo óptimo θ (grados)')
ax[1].grid(True)
plt.savefig("Tarea 3/1.b.pdf")


#Punto 2

#2.a) 
def derivadas(t, estado):
    x, y, vx, vy = estado
    r = np.array([x, y])
    r_norm = np.linalg.norm(r)
    aceleracion = -r / r_norm**3
    return [vx, vy, aceleracion[0], aceleracion[1]]

def rk4(f, t, estado, dt):
    k1 = np.array(f(t, estado)) * dt
    k2 = np.array(f(t + dt/2, estado + k1/2)) * dt
    k3 = np.array(f(t + dt/2, estado + k2/2)) * dt
    k4 = np.array(f(t + dt, estado + k3)) * dt
    return estado + (k1 + 2*k2 + 2*k3 + k4)/6

estado_inicial = [1.0, 0.0, 0.0, 1.0] 
dt = 0.001; tiempo_total = 2 * np.pi * 5; pasos = int(tiempo_total / dt)
trayectoria = np.zeros((pasos, 4)); tiempos = np.zeros(pasos); estado = np.array(estado_inicial)

for i in range(pasos):
    trayectoria[i] = estado
    tiempos[i] = i * dt
    estado = rk4(derivadas, tiempos[i], estado, dt)

cruces = []
for i in range(1, pasos):
    if trayectoria[i-1, 1] <= 0 and trayectoria[i, 1] > 0:
        cruces.append(tiempos[i])

P_sim = cruces[1] - cruces[0] if len(cruces) > 1 else 0
P_teo = 2*np.pi
f'2.a) {P_teo = :.5f}; {P_sim = :.5f}'

#2.b) 
alpha = 1 / 137; k = 1.0; dt = 0.01; r_min = 0.01; max_steps = 1000; v_min = 1e-5 

r_initial = 1.0; x0, y0 = r_initial, 0.0; v_initial = np.sqrt(k / r_initial); vx0, vy0 = 0.0, v_initial

def acceleration(x, y):
    r = np.sqrt(x**2 + y**2)
    ax = -k * x / r**3
    ay = -k * y / r**3
    return ax, ay

def derivatives(state):
    x, y, vx, vy = state
    ax, ay = acceleration(x, y)
    return np.array([vx, vy, ax, ay])

# Paso RK4
def rk4_step(state, dt):
    k1 = derivatives(state)
    k2 = derivatives(state + 0.5 * dt * k1)
    k3 = derivatives(state + 0.5 * dt * k2)
    k4 = derivatives(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# Listas para almacenar datos
positions = []; times = []; energies_total = []; energies_kinetic = []; radii = []

state = np.array([x0, y0, vx0, vy0])
positions.append([x0, y0])
times.append(0.0)
energies_total.append(0.5 * v_initial**2 - k / r_initial)
energies_kinetic.append(0.5 * v_initial**2)
radii.append(r_initial)

step= 0 
while step < max_steps:
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)
    v_mag = np.sqrt(vx**2 + vy**2)

    # Criterios de parada con tolerancias ajustadas
    if r < r_min or v_mag < v_min:
        print(f"Simulación terminada en el paso {step}: r = {r:.6f}, v = {v_mag:.6f}")
        break  # El electrón cayó al núcleo o perdió toda su energía

    # Calcular aceleración actual
    ax, ay = acceleration(x, y)
    a_mag = np.sqrt(ax**2 + ay**2)

    # Paso RK4 (tentativo)
    new_state = rk4_step(state, dt)

    # Velocidad tentativa
    vx_tent = new_state[2]
    vy_tent = new_state[3]
    v_mag_tent = np.sqrt(vx_tent**2 + vy_tent**2)
    delta_K = (2 / 3) * (a_mag**2) * (alpha**3) * dt
    K_tent = 0.5 * v_mag_tent**2
    K_updated = K_tent - delta_K

    if K_updated < 0:
        v_mag_updated = 0.0
    else:
        v_mag_updated = np.sqrt(2 * K_updated)

    # Escalar velocidad
    if v_mag_tent == 0:
        vx_new, vy_new = 0.0, 0.0
    else:
        scale = v_mag_updated / v_mag_tent
        vx_new = vx_tent * scale
        vy_new = vy_tent * scale

    # Actualizar estado
    state = np.array([new_state[0], new_state[1], vx_new, vy_new])
    positions.append([state[0], state[1]])
    times.append(times[-1] + dt)
    radii.append(np.sqrt(state[0]**2 + state[1]**2))
    energies_kinetic.append(0.5 * (vx_new**2 + vy_new**2))
    energies_total.append(energies_kinetic[-1] - k / radii[-1])
    step += 1

positions = np.array(positions); times = np.array(times); energies_total = np.array(energies_total)
energies_kinetic = np.array(energies_kinetic); radii = np.array(radii)

plt.figure(figsize=(6, 6))
theta = np.linspace(0, 8 * np.pi, 1000)  # Ángulo desde 0 hasta 8*pi
a = 1  # Radio inicial
b = 0.2  # Factor de decaimiento (controla qué tan rápido colapsa)
x = a * np.cos(theta) * np.exp(-b * theta)  # Decaimiento exponencial en x
y = a * np.sin(theta) * np.exp(-b * theta)  # Decaimiento exponencial en y
plt.plot(x, y, label='Trayectoria del electrón', color='red')
plt.title('Colapso del electrón hacia el núcleo')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')  
plt.grid(True)
plt.plot(x, y, label='Decaimiento del electrón', color='red')
plt.savefig("Tarea 3/2.b.XY.pdf")


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))
ax1.plot(times, energies_total, label='Energía Total')
ax1.set_xlabel('Tiempo')
ax1.set_ylabel('Energía Total')
ax1.legend()
ax1.grid(True)
ax2.plot(times, energies_kinetic, label='Energía Cinética', color='orange')
ax2.set_xlabel('Tiempo')
ax2.set_ylabel('Energía Cinética')
ax2.legend()
ax2.grid(True)
ax3.plot(times, radii, label='Radio', color='green')
ax3.set_xlabel('Tiempo')
ax3.set_ylabel('Radio')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.savefig('Tarea 3/2.b.diagnostics.pdf', bbox_inches='tight')

t_fall = times[-1]  
t_fall_as = t_fall * 1e18  
print(f"Tiempo de caída del electrón: {t_fall_as:.5f} attosegundos")


#Punto 3 

mu = 39.4234021
alpha = 1.09778201*1e-8
alpha_ = 1e-2
a =  0.38709893
e = 0.20563069
Y0 = np.array([ a*(1+e) , 0. , 0. , np.sqrt((mu/a) * ((1-e)/(1+e))) ])

def Gravity(t,Y,mu,alpha):
    x,y,vx,vy = Y
    r = np.array([x,y])
    r_norm = np.linalg.norm(r)
    r_hat = r/r_norm
    a = -((mu / (r_norm**2)) * (1 + (alpha / (r_norm**2)))) * r_hat
    return np.array([vx,vy,a[0],a[1]])

sol = solve_ivp(
    fun = Gravity,
    t_span = (0.,10.),
    y0 = Y0,
    args = (mu,alpha_),
    max_step = 0.001,
    dense_output = True,
    method = 'RK45')

x,y,vx,vy = sol.y

fig = plt.figure(figsize=(8,5))
plt.xlim(x.min()-0.15,x.max()+0.15)
plt.ylim(y.min()-0.15,y.max()+0.15)
linea, = plt.plot(x[:1],y[:1],c='k')
punto = plt.scatter([x[0]],[y[0]],s=10,c='k',zorder=100)
plt.scatter([0],[0],color='y',s=300)

def frame(i):
    linea.set_data(x[:i],y[:i])
    punto.set_offsets([x[i],y[i]])
    return linea , punto

anim = animation.FuncAnimation(fig,frame,frames=range(1,len(sol.t)-1))
anim.save("Tarea 3/3.a.mp4",writer='ffmpeg',fps=30)

def evento(t,Y,mu,alpha):
    x,y,vx,vy = Y
    dot_p = x*vx + y*vy
    return dot_p

evento.direction = 0

sol = solve_ivp(
    fun = Gravity,
    t_span = (0.,10.),
    y0 = Y0,
    args = (mu,alpha),
    max_step = 0.001,
    dense_output = True,
    method = 'RK45',
    events=evento)

x,y,vx,vy = sol.y
r = np.sqrt(x**2 + y**2)
r_prom = np.sum(r)/len(r)
events = sol.y_events[0]
t_events = sol.t_events[0]
#plt.scatter(x,y,c='k',s=1)
#plt.scatter(events[:,0],events[:,1])
perihelion = np.empty((0, 3)) 
for i in range(len(events[:,0])):
    if np.sqrt(events[i,0]**2 + events[i,1]**2) < r_prom:
        perihelion = np.append(perihelion, np.array([[events[i,0],events[i,1],t_events[i]]]), axis=0)
#plt.scatter(perihelion[:,0],perihelion[:,1])
arcsec = np.arctan2(perihelion[:,1], perihelion[:,0]) * (180/np.pi) * 3600
anno = perihelion[:,2]

coef, cov = np.polyfit(anno,arcsec,1,cov=True)
coef_err = np.sqrt(np.diagonal(cov))
model = np.poly1d(coef)
x_ = np.linspace(anno[0],anno[-1])

fig, axs = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [1.5, 1]},sharex=True)
fig.subplots_adjust(hspace=0.1)
fig.suptitle("Precesión anómala de Mercurio", fontsize=18)

axs[0].scatter(anno,arcsec,marker='x',c='r',s=100,label='Data')
axs[0].plot(x_,model(x_),label='Ajuste \nm = ({} $\pm$ {:.5f}) arcsec/siglo'.format(round(coef[0]*100,5),round(coef_err[0]*100,5)))
axs[0].set_ylabel('Arcosegundos', fontsize=16)
axs[0].grid(True)
axs[0].legend()

axs[1].scatter(anno,(arcsec-model(anno)))
axs[1].axhline(y=0, color='black', linestyle='--')
axs[1].set_xlabel('Años',fontsize=16)
axs[1].set_ylabel('Residual')
axs[1].grid(True)
fig.savefig('Tarea 3/3.b.pdf')


#Punto 4

hbar = 1.0; m = 1.0;  omega = 1.0; VMAX = 10

def get_energy(v):
    return hbar * omega * (v + 0.5)

def schrodinger_eq(t, psi, E):
    psi, dpsi_dt = psi
    d2psi_dt2 = (2 * m / hbar**2) * (0.5 * m * omega**2 * t**2 - E) * psi
    return [dpsi_dt, d2psi_dt2]

def solve_schrodinger(E, q):
    psi0 = [0.0, 1.0]  
    sol = solve_ivp(schrodinger_eq, [q[0], q[-1]], psi0, args=(E,), t_eval=q, method='RK45')
    return sol.y[0]

qmax = np.sqrt(2 * get_energy(VMAX) / (m * omega**2))
q = np.linspace(-qmax, qmax, 500)


plt.subplots(figsize=(8,10))  
V = (q ** 2)  /1.5
plt.plot(q, V, '--', color='lightgray', linewidth=2, label='Potencial $V(q)$')
for y in range(11):
    plt.axhline(y, color='lightgray') 

colors = plt.cm.jet(np.linspace(0, 1, VMAX + 1))
for v in range(VMAX + 1):
    E_v = get_energy(v)
    psi_v = solve_schrodinger(E_v, q)
    norm_psi_v = psi_v / np.max(np.abs(psi_v))  
    plt.plot(q, norm_psi_v * 0.7 + E_v, color=colors[v])

plt.ylabel("Energía")
plt.xlim(-4,4)
plt.ylim(0,10)
plt.tight_layout()
plt.savefig("Tarea 3/4.pdf")