import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp


# Punto 1

# Parámetros dados
n = 10
alpha = 4/5
num_samples = 500000
burn_in = 10000  
bins = 200


def g(x, n, alpha):
    return sum(np.exp(-(x - k)**2 * k) / (k**alpha) for k in range(1, n+1))

def metropolis_hastings(g, num_samples, proposal_std=1.0):
    samples = []
    x_current = np.random.uniform(0, n)  # Inicialización aleatoria
    acceptance_count = 0  # Para monitorear la tasa de aceptación
    
    for _ in range(burn_in):
        x_proposed = x_current + np.random.normal(0, proposal_std)
        if x_proposed < 0 or x_proposed > n:  # Rechazar propuestas fuera de [0, n]
            x_proposed = x_current
        acceptance_ratio = g(x_proposed, n, alpha) / g(x_current, n, alpha)
        if np.random.rand() < acceptance_ratio:
            x_current = x_proposed
            acceptance_count += 1
    
    for _ in range(num_samples):
        x_proposed = x_current + np.random.normal(0, proposal_std)
        if x_proposed < 0 or x_proposed > n:  # Rechazar propuestas fuera de [0, n]
            x_proposed = x_current
        acceptance_ratio = g(x_proposed, n, alpha) / g(x_current, n, alpha)
        if np.random.rand() < acceptance_ratio:
            x_current = x_proposed
            acceptance_count += 1
        samples.append(x_current)
    
    acceptance_rate = acceptance_count / (burn_in + num_samples)
    
    return np.array(samples)

samples = metropolis_hastings(g, num_samples)

plt.figure(figsize=(10, 6))
plt.hist(samples, bins=bins, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('x')
plt.ylabel('Densidad')
plt.title('Histograma de muestras generadas con Metrópolis-Hastings')
plt.grid(True)
plt.savefig("Tarea 4/1.a.pdf")

def f(x):
    return np.exp(-x**2)

integral_f = np.sqrt(np.pi)/2  
ratios = np.array([f(x) / g(x, n, alpha) for x in samples])
A_est = integral_f * num_samples / np.sum(ratios)
std_A = integral_f * np.std(ratios) / np.sqrt(num_samples)

A_teorico = sum(np.sqrt(np.pi) / (k**(alpha + 0.5)) for k in range(1, n+1))

print(f"1.b) A estimado: {A_est:.6f} ± {std_A:.6f}")
#print(f"1.b) A teórico: {A_teorico:.6f}")



#Punto 2

# Constantes
D1 = 50; D2 = 50; lambda_ = 670e-7; A = 0.04; a = 0.01; d = 0.1  


def classical_model(z, lambda_, d, a, D2):
    theta = np.arctan(z / D2)  # Ángulo theta
    term1 = np.cos((np.pi * d / lambda_) * np.sin(theta))**2  
    term2 = np.sinc((a / lambda_) * np.sin(theta))**2  
    return term1 * term2  # Intensidad clásica

z_values = np.linspace(-0.4, 0.4, 400) 

I_classical = classical_model(z_values, lambda_, d, a, D2)
I_classical_normalized = I_classical / np.max(I_classical)

N = 100000
def I(z):
    x_samples = np.random.uniform(-A/2, A/2, N)
    y_samples = np.random.uniform(-a/2, a/2, N)
    phase = (2 * np.pi / lambda_) * (D1 + D2) + \
            (np.pi / (lambda_ * D1)) * (x_samples - y_samples)**2 + \
            (np.pi / (lambda_ * D2)) * (z - y_samples)**2
    integral = np.sum(np.exp(1j * phase))
    return np.abs(integral)**2

I_values = np.array([I(z) for z in z_values])
I_normalized = I_values / np.max(I_values)


plt.style.use('dark_background')
plt.figure(figsize=(10, 6))
plt.plot(z_values, I_classical_normalized, label="Intensidad Clásica Normalizada", color="skyblue")
plt.plot(z_values, I_normalized, color="blue",label="Intensidad no Clásica Normalizada" , linewidth=2)
plt.xlabel("z (cm)")
plt.ylabel("Intensidad Normalizada")
plt.title("Intensidad Normalizada vs z")
plt.legend()
plt.savefig("Tarea 4/2.pdf")



#Punto 3

N = 150 
J = 0.2 
beta = 10 
iterations_per_frame = 400
frames = 500  

lattice = np.random.choice([-1, 1], size=(N, N))

def compute_energy(lattice, J):
    energy = 0
    for i in range(N):
        for j in range(N):
            energy -= J * lattice[i, j] * (lattice[(i + 1) % N, j] + lattice[i, (j + 1) % N])
    return energy

def metropolis_step(lattice, J, beta):
    
    i, j = np.random.randint(0, N, size=2)
    
    delta_E = 2 * J * lattice[i, j] * (
        lattice[(i + 1) % N, j] + lattice[i, (j + 1) % N] + lattice[(i - 1) % N, j] + lattice[i, (j - 1) % N]
    )
    
    if delta_E < 0 or np.random.rand() < np.exp(-beta * delta_E):
        lattice[i, j] *= -1
    
def update(frame_num, lattice, ax, J, beta):

    for _ in range(iterations_per_frame):
        metropolis_step(lattice, J, beta)
    
    ax.clear()
    ax.imshow(lattice, cmap='coolwarm', interpolation='nearest')
    ax.set_title(f"Frame {frame_num + 1}")
    ax.axis('off')

fig, ax = plt.subplots(figsize=(8, 8))
ani = animation.FuncAnimation(fig, update, frames=frames, fargs=(lattice, ax, J, beta), interval=50)

ani.save('Tarea 4/3.mp4', writer='ffmpeg', fps=30)



#Punto 5

# Constantes
A = 1000; B = 20; t_half_U = 23.4 / (60 * 24); t_half_Np = 2.36  
lambda_U = np.log(2) / t_half_U
lambda_Np = np.log(2) / t_half_Np

def system(t, y):
    U, Np, Pu = y
    dU_dt = A - lambda_U * U
    dNp_dt = lambda_U * U - lambda_Np * Np
    dPu_dt = lambda_Np * Np - B * Pu
    return [dU_dt, dNp_dt, dPu_dt]

y0 = [0, 0, 0]; t_span = (0, 30)

sol = solve_ivp(system, t_span, y0, t_eval=np.linspace(0, 30, 1000))

def find_stabilization_time(t, y, theoretical_value, tolerance=1e-2):
    for i in range(len(t)):
        if abs(y[i] - theoretical_value) / theoretical_value < tolerance:
            return t[i]
    return None

theoretical_U = A / lambda_U; theoretical_Np = A / lambda_Np; theoretical_Pu = A / B

stabilization_time_U = find_stabilization_time(sol.t, sol.y[0], theoretical_U)
stabilization_time_Np = find_stabilization_time(sol.t, sol.y[1], theoretical_Np)
stabilization_time_Pu = find_stabilization_time(sol.t, sol.y[2], theoretical_Pu)

"""print(f"5.a) ")
print(f"Tiempo de estabilización de U: {stabilization_time_U:.2f} días")
print(f"Tiempo de estabilización de Np: {stabilization_time_Np:.2f} días")
print(f"Tiempo de estabilización de Pu: {stabilization_time_Pu:.2f} días")"""

# 5.b) Simulación estocástica
R = np.array([
    [1, 0, 0],  # Creación de U-239
    [-1, 1, 0], # Decaimiento de U-239 a Np-239
    [0, -1, 1], # Decaimiento de Np-239 a Pu-239
    [0, 0, -1]  # Extracción de Pu-239
])

def calculate_rates(U, Np, Pu):
    return np.array([A, lambda_U * U, lambda_Np * Np, B * Pu])

def stochastic_simulation():
    Y = np.array([0, 0, 0]); t = 0; t_max = 30 
    time_points = [t]; U_values = [Y[0]]; Np_values = [Y[1]]; Pu_values = [Y[2]]

    while t < t_max:
        rates = calculate_rates(Y[0], Y[1], Y[2])
        total_rate = rates.sum()
        
        if total_rate > 0:
            tau = np.random.exponential(1 / total_rate)
            if t + tau > t_max:
                tau = t_max - t
            reaction = np.random.choice(len(rates), p=rates / total_rate)
            Y += R[reaction]
            t += tau
            time_points.append(t)
            U_values.append(Y[0])
            Np_values.append(Y[1])
            Pu_values.append(Y[2])
        else:
            t = t_max

    return time_points, U_values, Np_values, Pu_values

# 5.c) Plotear
num_simulations = 100
stochastic_U = []; stochastic_Np = []; stochastic_Pu = []

for _ in range(num_simulations):
    time_points, U_values, Np_values, Pu_values = stochastic_simulation()
    stochastic_U.append((time_points, U_values))
    stochastic_Np.append((time_points, Np_values))
    stochastic_Pu.append((time_points, Pu_values))


plt.style.use('dark_background')

fig, axes = plt.subplots(3, 2, figsize=(15, 15))

axes[0, 0].set_title("Método Determinista", color="white")
axes[0, 1].set_title("Método Estocástico", color="white")

color_U = "#FF6F61"
color_Np = "#6B5B95"
color_Pu = "#88B04B"

axes[0, 0].loglog(sol.t, sol.y[0], label="U", color=color_U, linewidth=2)
axes[0, 0].set_ylabel("Cantidad de U", color="white")
axes[0, 0].tick_params(axis='both', colors='white')

for i in range(num_simulations):
    axes[0, 1].plot(stochastic_U[i][0], stochastic_U[i][1], alpha=0.1, color=color_U)
axes[0, 1].set_ylabel("Cantidad de U", color="white")
axes[0, 1].tick_params(axis='both', colors='white')

axes[1, 0].plot(sol.t, sol.y[1], label="Np", color=color_Np, linewidth=2)
axes[1, 0].set_ylabel("Cantidad de Np", color="white")
axes[1, 0].tick_params(axis='both', colors='white')

for i in range(num_simulations):
    axes[1, 1].plot(stochastic_Np[i][0], stochastic_Np[i][1], alpha=0.1, color=color_Np)
axes[1, 1].set_ylabel("Cantidad de Np", color="white")
axes[1, 1].tick_params(axis='both', colors='white')

axes[2, 0].plot(sol.t, sol.y[2], label="Pu", color=color_Pu, linewidth=2)
axes[2, 0].set_ylabel("Cantidad de Pu", color="white")
axes[2, 0].set_xlabel("Tiempo (días)", color="white")
axes[2, 0].tick_params(axis='both', colors='white')

for i in range(num_simulations):
    axes[2, 1].plot(stochastic_Pu[i][0], stochastic_Pu[i][1], alpha=0.1, color=color_Pu)
axes[2, 1].set_ylabel("Cantidad de Pu", color="white")
axes[2, 1].set_xlabel("Tiempo (días)", color="white")
axes[2, 1].tick_params(axis='both', colors='white')

plt.tight_layout()
plt.savefig("Tarea 4/5.pdf")

#5.d) 
num_simulations_large = 1000
count_Pu_above_80 = 0  

for _ in range(num_simulations_large):
    _, _, _, Pu_values = stochastic_simulation()
    if max(Pu_values) >= 80: 
        count_Pu_above_80 += 1

probability_Pu_above_80 = round(count_Pu_above_80 / num_simulations_large,2)
print(f"5) Probabilidad de que Pu sea mayor 80%: {probability_Pu_above_80:.4f}")
"""
Una opción efectiva es aumentar la tasa de extracción de Plutonio (B), lo que aceleraría la eliminación de Pu del sistema y reduciría su acumulación. 
Otra alternativa es disminuir la tasa de producción de Uranio (A), limitando así la cantidad de material disponible para decaer en Neptunio y luego en Plutonio. 
También se podría aumentar la tasa de decaimiento de Uranio (λ_U) o de Neptunio (λ_Np), lo que aceleraría la transformación de estos elementos y podría ayudar a controlar los niveles de Pu. 
Estos cambios deben implementarse de manera equilibrada para no afectar negativamente otras partes del sistema.
"""