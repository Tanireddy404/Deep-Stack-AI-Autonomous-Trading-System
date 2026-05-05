import os, copy, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from collections import deque
from deepstack_core import COINS, COIN_LABELS, LOOKBACK_WINDOW, N_FEATURES, MultiCoinTradingEnv, fetch_data

EPISODES        = 1000
BATCH_SIZE      = 256
GAMMA           = 0.99
LR              = 2e-4
REPLAY_CAPACITY = 20_000
TAU             = 0.02
EPSILON_START   = 1.0
EPSILON_END     = 0.05
EPSILON_DECAY   = 0.990
EARLY_STOP      = 500
SAVE_DIR        = 'models'
os.makedirs(SAVE_DIR, exist_ok=True)

N_COINS   = len(COINS)
OBS_H     = LOOKBACK_WINDOW
OBS_W     = N_COINS * N_FEATURES
N_ACTIONS = 3 ** N_COINS
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training on: {DEVICE}')

class DeepStackNet(nn.Module):
    def __init__(self, obs_h, obs_w, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(obs_w, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),    nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, cap): self.buf = deque(maxlen=cap)
    def push(self, *t): self.buf.append(t)
    def sample(self, n):
        batch = random.sample(self.buf, n)
        s,a,r,s2,d = zip(*batch)
        return (torch.tensor(np.array(s),  dtype=torch.float32).to(DEVICE),
                torch.tensor(a,            dtype=torch.long).to(DEVICE),
                torch.tensor(r,            dtype=torch.float32).to(DEVICE),
                torch.tensor(np.array(s2), dtype=torch.float32).to(DEVICE),
                torch.tensor(d,            dtype=torch.float32).to(DEVICE))
    def __len__(self): return len(self.buf)

def flat_to_multi(flat, n=N_COINS):
    a = []
    for _ in range(n): a.append(flat % 3); flat //= 3
    return list(reversed(a))

def train():
    from csv_loader import fetch_data_csv
    print('Loading data from CSV...')
    data = fetch_data_csv(COINS)
    env  = MultiCoinTradingEnv(data, COINS)

    policy_net = DeepStackNet(OBS_H, OBS_W, N_ACTIONS).to(DEVICE)
    target_net = copy.deepcopy(policy_net).to(DEVICE)
    target_net.eval()
    optimizer  = optim.Adam(policy_net.parameters(), lr=LR)
    buffer     = ReplayBuffer(REPLAY_CAPACITY)
    epsilon    = EPSILON_START
    best_reward, patience = -np.inf, 0

    for episode in range(1, EPISODES+1):
        state = env.reset(); ep_reward = 0.0; done = False
        while not done:
            if random.random() < epsilon:
                flat_a = random.randrange(N_ACTIONS)
            else:
                with torch.no_grad():
                    s_t    = torch.tensor(state[None], dtype=torch.float32).to(DEVICE)
                    flat_a = policy_net(s_t).argmax(1).item()
            next_state, reward, done, info = env.step(flat_to_multi(flat_a))
            buffer.push(state, flat_a, reward, next_state, float(done))
            state = next_state; ep_reward += reward
            if len(buffer) >= BATCH_SIZE:
                s,a,r,s2,d = buffer.sample(BATCH_SIZE)
                q = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    tgt = r + GAMMA * target_net(s2).max(1)[0] * (1-d)
                loss = nn.SmoothL1Loss()(q, tgt)
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
                for tp,pp in zip(target_net.parameters(), policy_net.parameters()):
                    tp.data.copy_(TAU*pp.data + (1-TAU)*tp.data)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        if episode % 20 == 0:
            print(f'Ep {episode:4d}/{EPISODES} | Reward: {ep_reward:+.4f} | Portfolio: ${float(info["portfolio_value"]):,.2f} | e: {epsilon:.3f}')
        if ep_reward > best_reward:
            best_reward = ep_reward; patience = 0
            torch.save({'episode':episode,'model_state_dict':policy_net.state_dict(),'reward':best_reward,'coins':COINS}, f'{SAVE_DIR}/deepstack_best_model.pth')
            print(f'  Best model saved! ep={episode} reward={best_reward:.4f}')
        else:
            patience += 1
        if patience >= EARLY_STOP:
            print(f'Early stop at episode {episode}'); break
    print(f'Done! Best reward: {best_reward:.4f}')

if __name__ == '__main__':
    train()
