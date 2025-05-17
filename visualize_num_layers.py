import matplotlib.pyplot as plt

# Data from the table
num_layers = [2,3,4,5,6,7,8]
bleu4 = [0.280,0.298,0.307,0.285,0.261,0.198,0.199]
rouge = [0.448,0.452,0.452,0.436,0.408,0.429,0.427]
meteor = [0.243,0.251,0.253,0.239,0.225,0.211,0.209]

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
