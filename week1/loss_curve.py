from neuron import neuron
from neuron import relu, sigmoid
from loss import squared_error 
from numpy import arange 
import matplotlib.pyplot as plt   # type: ignore
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
 
def plot_loss_curve(x: float, bias: float, y_true: float, activation): 
    w_values: list = []
    losses:list  = []
    for w in arange(-10, 10, 0.1): 
        y_hat = neuron([x], [w], bias, activation)
        loss = squared_error(y_hat, y_true)
        w_values.append(w)
        losses.append(loss)

    plt.plot(w_values, losses)
    plt.title("Ağırlık Parametrelerine Göre Loss Değişimi")
    plt.xlabel("w")
    plt.ylabel("loss")
    plt.savefig(f"plots/loss_curve_{timestamp}.png")
    plt.show()
    

plot_loss_curve(x=5, bias=-1.5, y_true=-10, activation=relu)