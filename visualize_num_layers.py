import matplotlib.pyplot as plt

# Data from the table
num_layers = [3, 5, 7]
bleu4 = [0.298, 0.285, 0.198]
rouge = [0.452, 0.436, 0.429]
meteor = [0.251, 0.239, 0.211]

# Plotting the lines
plt.figure(figsize=(8, 5))
plt.plot(num_layers, bleu4, marker='o', label='BLEU@4', color='blue')
plt.plot(num_layers, rouge, marker='s', label='ROUGE', color='green')
plt.plot(num_layers, meteor, marker='^', label='METEOR', color='red')

# Adding labels and title
plt.xlabel('Number of Layers')
plt.ylabel('Score')
plt.title('Performance Metrics vs. Number of Layers')
plt.legend()
plt.grid(True)
plt.xticks(num_layers)
plt.tight_layout()

# Show plot
plt.show()
