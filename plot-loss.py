import matplotlib.pyplot as plt

def plot_loss(loss):
    epochs = range(1, len(loss) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss, marker='o', linestyle='-', linewidth=2, markersize=6)
    plt.title("Training Loss Curve", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


loss = [
    0.7890, 0.5967, 0.5599, 0.5429, 0.5481,
    0.5229, 0.5114, 0.4991, 0.4906, 0.4884,
    0.4900, 0.4628, 0.4712, 0.4613, 0.4558,
    0.4442, 0.4397, 0.4476, 0.4372, 0.4354
]

plot_loss(loss)
