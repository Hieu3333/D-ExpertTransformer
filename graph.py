import matplotlib.pyplot as plt
import numpy as np

# Data
metrics = ['B@1', 'B@2', 'B@3', 'B@4', 'ROUGE', 'METEOR']
with_keywords = [0.526, 0.435, 0.376, 0.330, 0.501, 0.252]
without_keywords = [0.377, 0.272, 0.213, 0.169, 0.379, 0.160]

# Bar width and positions
x = np.arange(len(metrics))
width = 0.35

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, with_keywords, width, label='With keywords', color='#4C72B0')
bars2 = ax.bar(x + width/2, without_keywords, width, label='Without keywords', color='#DD8452')

# Labels and titles
ax.set_xlabel('Metrics', fontsize=12)
ax.set_ylabel('Scores', fontsize=12)
ax.set_title('', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.legend()

# Adding value labels on bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

add_labels(bars1)
add_labels(bars2)

plt.tight_layout()
plt.show()
