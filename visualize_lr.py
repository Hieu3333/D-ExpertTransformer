import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR

# Configuration
epochs = 50
cnn_lr_max = 0.0001
model_lr_max = 0.0002
min_lr = 1e-6

# Create dummy optimizers
model = torch.nn.Module()  # Dummy model
cnn_params = [torch.nn.Parameter(torch.randn(2, 2))]  # CNN base parameters
model_params = [torch.nn.Parameter(torch.randn(2, 2))]  # Other parameters

# Create separate optimizers for clear visualization
cnn_optimizer = optim.AdamW(cnn_params, lr=cnn_lr_max)
model_optimizer = optim.AdamW(model_params, lr=model_lr_max)

# Create schedulers
cnn_scheduler = CosineAnnealingLR(cnn_optimizer, T_max=epochs, eta_min=min_lr)
model_scheduler = CosineAnnealingLR(model_optimizer, T_max=epochs, eta_min=min_lr)

# Store learning rates
cnn_lrs = []
model_lrs = []

for epoch in range(epochs):
    # Record before step
    cnn_lrs.append(cnn_optimizer.param_groups[0]['lr'])
    model_lrs.append(model_optimizer.param_groups[0]['lr'])
    
    # Update (simulate training step)
    cnn_scheduler.step()
    model_scheduler.step()

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(range(epochs), cnn_lrs, 'b-', label=f'CNN Base (max={cnn_lr_max})', linewidth=2)
plt.plot(range(epochs), model_lrs, 'r-', label=f'Remainder of Model (max={model_lr_max})', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Learning Rate', fontsize=12)
plt.title('Model Learning Rate \n(Min LR={})'.format(min_lr), fontsize=14)
plt.yscale('log')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.xticks(range(0, epochs+1, 5))
plt.tight_layout()
plt.show()