import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

output_path = r"D:\Reforinment-Learing-in-Advesararial-Attack-with-Image-Classification-Model\rewards (1).json"

def plot_rewards(rewards_all, window=2000):
    smoothed_y = savgol_filter(rewards_all, window_length=500, polyorder=2, mode='nearest')

    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(rewards_all)), smoothed_y, color='magenta')
    plt.grid(True)
    plt.xlabel('Steps')
    plt.ylabel('Reward')

    # Adjust y-axis limits to focus on the range of the rewards
    min_reward = min(rewards_all)
    max_reward = max(rewards_all)
    padding = 0.1 * (max_reward - min_reward)
    plt.ylim(min_reward - padding, max_reward + padding)

    plt.legend()
    plt.show()

with open(output_path, "r") as f:
    l = json.load(f)

new_l = [l[0]]
for i in range(1, len(l)):
    new_l.append(l[i - 1] + l[i])

print(min(new_l))
print(max(new_l))

plot_rewards(new_l, window=2000)