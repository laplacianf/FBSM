import numpy as np
from matplotlib import pyplot

t = 500
n = 10**4
dt = t/n

A = 2900
B = 1500

beta = 0.29
mu = 0.16
sigma = 0.2

# no control
def nb(S, E, I, R):
    N = S + E + I + R
    x = [0, 0, 0, 0]
    x[0] = -beta * I * S / N # S
    x[1] = beta * I * S / N - sigma * E # E
    x[2] = sigma * E - mu * I # I
    x[3] = mu * I # R
    return x

x = [np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)] # S E I R
x[0][0] = 4850 # S0
x[1][0] = 10 # E0
x[2][0] = 10 # I0
x[3][0] = 0 # R0

# RK4 forward
for i in range(1, n):
    S = (x[0][i] + x[0][i - 1])/2
    E = (x[1][i] + x[1][i - 1])/2
    I = (x[2][i] + x[2][i - 1])/2
    R = (x[3][i] + x[3][i - 1])/2

    # k1
    k1 = nb(S, E, I, R)
    k1_S = k1[0]
    k1_E = k1[1]
    k1_I = k1[2]
    k1_R = k1[3]

    # k2
    k2 = nb(S + k1_S * dt/2 , E + k1_E * dt/2 , I + k1_I * dt/2 , R + k1_R * dt/2)
    k2_S = k2[0]
    k2_E = k2[1]
    k2_I = k2[2]
    k2_R = k2[3]

    # k3
    k3 = nb(S + k2_S * dt/2 , E + k2_E * dt/2 , I + k2_I * dt/2 , R + k2_R * dt/2)
    k3_S = k3[0]
    k3_E = k3[1]
    k3_I = k3[2]
    k3_R = k3[3]

    # k4
    k4 = nb(S + k3_S * dt , E + k3_E * dt , I + k3_I * dt , R + k3_R * dt)
    k4_S = k4[0]
    k4_E = k4[1]
    k4_I = k4[2]
    k4_R = k4[3]

    x[0][i] = x[0][i - 1] + dt/6 * (k1_S + 2 * k2_S + 2 * k3_S + k4_S) # S
    x[1][i] = x[1][i - 1] + dt/6 * (k1_E + 2 * k2_E + 2 * k3_E + k4_E) # E
    x[2][i] = x[2][i - 1] + dt/6 * (k1_I + 2 * k2_I + 2 * k3_I + k4_I) # I
    x[3][i] = x[3][i - 1] + dt/6 * (k1_R + 2 * k2_R + 2 * k3_R + k4_R) # R

nS = x[0]
nE = x[1]
nI = x[2]
nR = x[3]

# control
def b(S, E, I, R, v, q):
    N = S + E + I + R
    x = [0, 0, 0, 0]
    x[0] = -beta * I * S / N - v * S # S
    x[1] = beta * I * S / N - sigma * E # E
    x[2] = sigma * E - mu * I - q * I # I
    x[3] = mu * I + q * I + v * S # R
    return x

def Hx(S, E, I, R, v, q, lS, lE, lI, lR):
    N = S + E + I + R
    x = [0, 0, 0, 0]
    x[0] = lS * (beta * I / N + v) - lE * beta * I / N - lR * v # lS
    x[1] = lE * sigma - lI * sigma # lE
    x[2] = -1 + lS * beta * S / N - lE * beta * S / N + lI * (mu + q) - lR * (mu + q) # lI
    x[3] = 0 # lR
    return x

u = [np.zeros(n), np.zeros(n)] # v q
omega = 0.5

for k in range(20):
    print('k =', k)
    x = [np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)] # S E I R
    x[0][0] = 4850 # S0
    x[1][0] = 10 # E0
    x[2][0] = 10 # I0
    x[3][0] = 0 # R0

    l = [np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)] # lS lE lI lR
    # terminal values
    l[0][n - 1] = 0 
    l[1][n - 1] = 0
    l[2][n - 1] = 0
    l[3][n - 1] = 0

    # RK4 forward
    for i in range(1, n):
        S = (x[0][i] + x[0][i - 1])/2
        E = (x[1][i] + x[1][i - 1])/2
        I = (x[2][i] + x[2][i - 1])/2
        R = (x[3][i] + x[3][i - 1])/2

        v = (u[0][i] + u[0][i - 1])/2
        q = (u[1][i] + u[1][i - 1])/2

        # k1
        k1 = b(S, E, I, R, v, q)
        k1_S = k1[0]
        k1_E = k1[1]
        k1_I = k1[2]
        k1_R = k1[3]

        # k2
        k2 = b(S + k1_S * dt/2 , E + k1_E * dt/2 , I + k1_I * dt/2 , R + k1_R * dt/2, v, q)
        k2_S = k2[0]
        k2_E = k2[1]
        k2_I = k2[2]
        k2_R = k2[3]

        # k3
        k3 = b(S + k2_S * dt/2 , E + k2_E * dt/2 , I + k2_I * dt/2 , R + k2_R * dt/2, v, q)
        k3_S = k3[0]
        k3_E = k3[1]
        k3_I = k3[2]
        k3_R = k3[3]

        # k4
        k4 = b(S + k3_S * dt , E + k3_E * dt , I + k3_I * dt, R + k3_R * dt, v, q)
        k4_S = k4[0]
        k4_E = k4[1]
        k4_I = k4[2]
        k4_R = k4[3]

        x[0][i] = x[0][i - 1] + dt/6 * (k1_S + 2 * k2_S + 2 * k3_S + k4_S) # S
        x[1][i] = x[1][i - 1] + dt/6 * (k1_E + 2 * k2_E + 2 * k3_E + k4_E) # E
        x[2][i] = x[2][i - 1] + dt/6 * (k1_I + 2 * k2_I + 2 * k3_I + k4_I) # I
        x[3][i] = x[3][i - 1] + dt/6 * (k1_R + 2 * k2_R + 2 * k3_R + k4_R) # R

    # RK4 backward
    for i in range(n - 1, 0, -1):
        S = (x[0][i] + x[0][i - 1])/2
        E = (x[1][i] + x[1][i - 1])/2
        I = (x[2][i] + x[2][i - 1])/2
        R = (x[3][i] + x[3][i - 1])/2

        v = (u[0][i] + u[0][i - 1])/2
        q = (u[1][i] + u[1][i - 1])/2

        lS = (l[0][i] + l[0][i - 1])/2
        lE = (l[1][i] + l[1][i - 1])/2
        lI = (l[2][i] + l[2][i - 1])/2
        lR = (l[3][i] + l[3][i - 1])/2

        # k1
        k1 = Hx(S, E, I, R, v, q, lS, lE, lI, lR)
        k1_lS = k1[0]
        k1_lE = k1[1]
        k1_lI = k1[2]
        k1_lR = k1[3]

        # k2
        k2 = Hx(S, E, I, R, v, q, lS - k1_lS * dt/2, lE - k1_lE * dt/2, lI - k1_lI * dt/2, lR - k1_lR * dt/2)
        k2_lS = k2[0]
        k2_lE = k2[1]
        k2_lI = k2[2]
        k2_lR = k2[3]

        # k3
        k3 = Hx(S, E, I, R, v, q, lS - k2_lS * dt/2, lE - k2_lE * dt/2, lI - k2_lI * dt/2, lR - k2_lR * dt/2)
        k3_lS = k3[0]
        k3_lE = k3[1]
        k3_lI = k3[2]
        k3_lR = k3[3]

        # k4
        k4 = Hx(S, E, I, R, v, q, lS - k3_lS * dt, lE - k3_lE * dt, lI - k3_lI * dt, lR - k3_lR * dt)
        k4_lS = k4[0]
        k4_lE = k4[1]
        k4_lI = k4[2]
        k4_lR = k4[3]

        l[0][i - 1] = l[0][i] - dt/6 * (k1_lS + 2 * k2_lS + 2 * k3_lS + k4_lS) # lS
        l[1][i - 1] = l[1][i] - dt/6 * (k1_lE + 2 * k2_lE + 2 * k3_lE + k4_lE) # lE
        l[2][i - 1] = l[2][i] - dt/6 * (k1_lI + 2 * k2_lI + 2 * k3_lI + k4_lI) # lI
        l[3][i - 1] = l[3][i] - dt/6 * (k1_lR + 2 * k2_lR + 2 * k3_lR + k4_lR) # lR
    
    # update v, q
    S = x[0]
    I = x[2]
    lS = l[0]
    lI = l[2]
    lR = l[3]

    u[0] = omega * u[0] + (1 - omega) * (lS * S - lR * S) / A # v
    u[1] = omega * u[1] + (1 - omega) * (lI * I - lR * I) / B # q

S = x[0]
E = x[1]
I = x[2]
R = x[3]

ts = np.linspace(0, t, n)

fig, axs = pyplot.subplots(2, 2)
axs[0, 0].plot(ts, S)
axs[0, 0].plot(ts, nS)
axs[0, 0].set(xlabel='$t$ (days)', ylabel='$S(t)$ (person)')
axs[0, 0].legend(('control', 'no control'))

axs[0, 1].plot(ts, E)
axs[0, 1].plot(ts, nE)
axs[0, 1].set(xlabel='$t$ (days)', ylabel='$E(t)$ (person)')
axs[0, 1].legend(('control', 'no control'))

axs[1, 0].plot(ts, I)
axs[1, 0].plot(ts, nI)
axs[1, 0].set(xlabel='$t$ (days)', ylabel='$I(t)$ (person)')
axs[1, 0].legend(('control', 'no control'))

axs[1, 1].plot(ts, R)
axs[1, 1].plot(ts, nR)
axs[1, 1].set(xlabel='$t$ (days)', ylabel='$R(t)$ (person)')
axs[1, 1].legend(('control', 'no control'))

pyplot.show()
