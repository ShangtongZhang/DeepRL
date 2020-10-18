import numpy as np
import matplotlib.pyplot as plt

NUM_STATES = 4
P_MU = 0.6 # Go clockwise

V_BAR = 0

def simulate_trajectory(max_steps):
    G = 0
    Gs = []
    state = np.random.randint(NUM_STATES)
    for _ in range(max_steps):
        action = np.random.choice([-1, 1], p=[1 - P_MU, P_MU])
        reward = 1
        next_state = (state + action + NUM_STATES) % NUM_STATES
        gamma_s = (0 if state == 3 else 1)
        G = reward + gamma_s * G
        Gs.append([next_state, G])
        state = next_state
    return Gs


def simulate_oracle():
    runs = 1000
    max_steps = 100
    data = dict()
    for s in range(NUM_STATES):
        for t in range(max_steps):
            data[(s, t)] = []
    for _ in range(runs):
        Gs = simulate_trajectory(max_steps)
        for t in range(max_steps):
            s, G = Gs[t]
            data[(s, t)].append(G)
    mean = np.zeros((NUM_STATES, max_steps))
    std = np.zeros(mean.shape)
    for s in range(NUM_STATES):
        for t in range(max_steps):
            mean[s, t] = np.mean(data[(s, t)])
            std[s, t] = np.std(data[(s, t)])
    state_to_plot = 0
    plt.errorbar(x=np.arange(mean.shape[1]), y=mean[state_to_plot], yerr=std[state_to_plot])
    plt.ylim([0, 10])
    plt.show()
    # print(np.mean(mean[:, -10:], axis=1))
    return data


def compute_oracle():
    P_pi = np.zeros((NUM_STATES, NUM_STATES))
    P_pi[0, 1] = P_pi[1, 2] = P_pi[2, 3] = P_pi[3, 0] = P_MU
    P_pi[0, 3] = P_pi[1, 0] = P_pi[2, 1] = P_pi[3, 2] = 1 - P_MU
    gamma = np.zeros(P_pi.shape)
    gamma[0, 0] = gamma[1, 1] = gamma[2, 2] = 1
    P_tilde = np.zeros((NUM_STATES * 2, NUM_STATES))
    for i in range(NUM_STATES):
        P_tilde[2 * i, (i + 1) % NUM_STATES] = 1
        P_tilde[2 * i +1, (i - 1 + NUM_STATES) % NUM_STATES] = 1
    r = np.ones((NUM_STATES * 2, 1))

    P = np.eye(NUM_STATES)
    P_star = 0
    N = 1000
    for _ in range(N):
        P_star += P
        P = P @ P_pi
    P_star = P_star / float(N)
    d = P_star[0, :]
    D_pi = np.diag(d)
    D_tilde = np.zeros((NUM_STATES * 2, NUM_STATES * 2))
    for i in range(NUM_STATES):
        D_tilde[2 * i, 2 * i] = d[i] * P_MU
        D_tilde[2 * i + 1, 2 * i + 1] = d[i] * (1 - P_MU)

    v_bar = np.linalg.inv(D_pi) @ np.linalg.inv(np.eye(NUM_STATES) - P_pi.T @ gamma) \
            @ P_tilde.T @ D_tilde @ r

    print(v_bar)
    return v_bar


if __name__ == '__main__':
    compute_oracle()
    simulate_oracle()
