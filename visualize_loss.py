import matplotlib.pyplot as plt
import numpy as np

# Epochs
epochs = np.arange(1, 51)

# Loss values
losses_effnet = [
    4.7187, 2.9661, 2.0436, 1.4962, 1.1295, 0.8747, 0.6844, 0.5456, 0.4316, 0.3481,
    0.2809, 0.2286, 0.1882, 0.1514, 0.1257, 0.1062, 0.0870, 0.0722, 0.0616, 0.0528,
    0.0450, 0.0392, 0.0347, 0.0305, 0.0265, 0.0244, 0.0215, 0.0205, 0.0193, 0.0178,
    0.0170, 0.0158, 0.0151, 0.0144, 0.0135, 0.0133, 0.0124, 0.0125, 0.0123, 0.0123,
    0.0117, 0.0114, 0.0111, 0.0108, 0.0108, 0.0107, 0.0107, 0.0107, 0.0105, 0.0105
]

losses_densenet = [
    5.3070, 4.1836, 3.3783, 2.8079, 2.3918, 2.0716, 1.8152, 1.6017, 1.4255, 1.2690,
    1.1368, 1.0270, 0.9168, 0.8297, 0.7511, 0.6716, 0.6103, 0.5602, 0.5059, 0.4580,
    0.4190, 0.3857, 0.3527, 0.3252, 0.2965, 0.2754, 0.2498, 0.2332, 0.2142, 0.1990,
    0.1853, 0.1740, 0.1636, 0.1536, 0.1440, 0.1361, 0.1297, 0.1232, 0.1179, 0.1135,
    0.1081, 0.1048, 0.1017, 0.0987, 0.0967, 0.0939, 0.0918, 0.0916, 0.0908, 0.0913
]

losses_resnet = [
    5.3127, 4.2123, 3.4129, 2.8585, 2.4572, 2.1508, 1.9054, 1.7029, 1.5294, 1.3739,
    1.2543, 1.1435, 1.0429, 0.9637, 0.8782, 0.8171, 0.7505, 0.6947, 0.6421, 0.5980,
    0.5528, 0.5123, 0.4680, 0.4349, 0.4026, 0.3747, 0.3456, 0.3228, 0.2982, 0.2808,
    0.2562, 0.2429, 0.2284, 0.2115, 0.2029, 0.1904, 0.1785, 0.1684, 0.1624, 0.1560,
    0.1494, 0.1438, 0.1381, 0.1362, 0.1312, 0.1290, 0.1288, 0.1267, 0.1243, 0.1236
]

# Plot
plt.figure(figsize=(12, 6))

# Plot each model's loss
plt.plot(epochs, losses_effnet, label='EfficientNet', linewidth=2)
plt.plot(epochs, losses_densenet, label='DenseNet', linewidth=2)
plt.plot(epochs, losses_resnet, label='ResNet', linewidth=2)

# Highlight final losses
plt.scatter(50, losses_effnet[-1], color='blue', s=70, zorder=5)
plt.scatter(50, losses_densenet[-1], color='orange', s=70, zorder=5)
plt.scatter(50, losses_resnet[-1], color='green', s=70, zorder=5)

# Annotations
# plt.annotate(f'{losses_effnet[-1]:.4f}', (50, losses_effnet[-1]), textcoords="offset points", xytext=(-30, 10), ha='center')
# plt.annotate(f'{losses_densenet[-1]:.4f}', (50, losses_densenet[-1]), textcoords="offset points", xytext=(-30, -15), ha='center')
# plt.annotate(f'{losses_resnet[-1]:.4f}', (50, losses_resnet[-1]), textcoords="offset points", xytext=(-30, -35), ha='center')

# Visual aesthetics
plt.title('Training Loss over Epochs for Different CNNs as visual extractor', fontsize=14, pad=20)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.yscale('log')  # Log scale for better visibility
plt.xticks(np.arange(0, 51, 5))
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10, loc='upper right')
plt.tight_layout()

# Show plot
plt.show()
