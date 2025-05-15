



import matplotlib.pyplot as plt
import numpy as np

# Labels (X-axis)
lambda_labels = [
    r'$\lambda = 0.3$',
    r'$\lambda = 0.5$',
    r'$\lambda = 0.8$'
]

# Scores from the table
bleu4 = [ 0.296, 0.287, 0.258]
rouge = [ 0.453, 0.451, 0.437]
meteor = [0.250, 0.248, 0.232]

# Position settings
x = np.arange(len(lambda_labels))
width = 0.25  # Width of each bar

# Create the bar chart
plt.figure(figsize=(10, 6))
plt.bar(x - width, bleu4, width, label='BLEU@4', color='skyblue')
plt.bar(x, rouge, width, label='ROUGE', color='mediumseagreen')
plt.bar(x + width, meteor, width, label='METEOR', color='salmon')

# Labeling
plt.xlabel(r'$\lambda$')
plt.ylabel('Score')
plt.title(r'Effect of $\lambda$ on Evaluation Metrics')
plt.xticks(x, lambda_labels)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show plot
plt.show()
