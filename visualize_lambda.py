import matplotlib.pyplot as plt
import numpy as np

# Data from the table
strategies = ["Layer-dependent", r"$\lambda_{init}$ = 0.3", r"$\lambda_{init}$ = 0.5", r"$\lambda_{init}$ = 0.8"]
metrics = [ "B@4", "ROUGE", "METEOR"]
values = [
    [ 0.298, 0.452, 0.251],
    [ 0.296, 0.453, 0.250],
    [ 0.287, 0.451, 0.248],
    [0.258, 0.437, 0.232]
]

# Plot settings
bar_width = 0.2
x = np.arange(len(metrics))

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['royalblue', 'cornflowerblue', 'deepskyblue', 'lightblue']

for i, (strategy, color) in enumerate(zip(strategies, colors)):
    ax.bar(x + i * bar_width, values[i], width=bar_width, label=strategy, color=color)

# Aesthetics
ax.set_xlabel("Metrics")
ax.set_ylabel("Scores")
ax.set_title("Results on various λ initialization strategies")
ax.set_xticks(x + bar_width * 1.5)
ax.set_xticklabels(metrics)
ax.set_ylim(0.2,0.5)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
