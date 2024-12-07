"""Note: dataset not uploaded to Gradescope because of size. """

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
from model import FullyConnectedNetwork_E, FullyConnectedNetwork_Psi
from data_loader import Dataset_Random_Potentials
from loss_function import ComplexMSELoss
import pickle

def test_1_epoch(dataloader, target_type, model, loss_fn):
    """ Testing a model on a dataset for 1 epoch

    Args:
        dataloader: iterable with batches of data 
        model: PyTorch model to train
        loss_fn: loss function

    Returns:
        float: accuracy
        float: total_loss
    """
    error = []
    model.eval()
    average_loss = 0
    with torch.no_grad():
        for V, E, Psi in dataloader:
            target = E if target_type=='E' else Psi
            V, target = V.float().to(device), target.to(device)
            pred = model(V)
            error.append(((pred.abs()-target.abs())/target.abs()).numpy())
            average_loss += loss_fn(torch.abs(pred), torch.abs(target)).item()
            print(average_loss)

    # if target_type=='Psi':
    #     # Visualize histogram of percent errors between nodes in target vs nodes in pred
    #     plt.title("Error histogram")
    #     plt.hist([p for i in error for j in i for k in j for p in k if not np.isinf(p)], bins=np.logspace(0,3,100), log=True, density=True)
    #     plt.xscale('log')
    #     plt.xlabel('Percent Error')
    #     plt.ylabel('Probability Density')
    #     plt.show()

    average_loss /= len(dataloader)
    print(f'Average Test Loss: {average_loss} \n')

if __name__ == '__main__':
    device = ("cpu")#"mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using {device} device")

    # Load Models
    model_Psi = FullyConnectedNetwork_Psi()
    model_Psi.load_state_dict(torch.load('models/model_2024-12-04_Psi', weights_only=True, map_location=torch.device('cpu')))
    model_Psi.eval()
    model_E = FullyConnectedNetwork_E()
    model_E.load_state_dict(torch.load('models/model_2024-12-04_E', weights_only=True, map_location=torch.device('cpu')))
    model_E.eval()

    # Load Loss as a Function of Epoch
    losses_Psi = []
    losses_E = []
    with open('models/e_loss.pkl', 'rb') as f:
        losses_E += pickle.load(f)[::2][0:25]
    with open('models/psi_loss.pkl', 'rb') as f:
        losses_Psi += pickle.load(f)[::2]


    # Load Data
    batch_size = 64
    epochs = 20
    loss_fn_complex = ComplexMSELoss()
    training_data = Dataset_Random_Potentials(load='data/randomly_generated_dataset.pkl')
    train_set, val_set = random_split(training_data, [70000, 10000], generator=torch.Generator().manual_seed(1))
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # MSE on validation set
    test_1_epoch(dataloader=test_dataloader, target_type='Psi', model=model_Psi, loss_fn=ComplexMSELoss())
    test_1_epoch(dataloader=test_dataloader, target_type='E', model=model_E, loss_fn=torch.nn.MSELoss())

    # Generate plots for random datapoints
    for i in range(len(val_set)):
        # Calculate predicted Psi
        V, E, Psi = val_set[i]
        V = torch.tensor([V]).to(device)
        predicted = model_Psi(V)

        # Plot actual vs predicted psi
        x = Psi
        y = predicted.detach().numpy()[0]
        plt.figure(2)
        plt.subplot(132)
        plt.plot(np.abs(x[:,0]),label='Actual', color='red')
        plt.plot(np.abs(y[:,0]),label='Predicted', color='blue')
        plt.legend()
        plt.ylabel('Wavefunction')
        plt.xlabel('x')
        plt.title(f'Error = {ComplexMSELoss()(torch.Tensor(np.abs(x[:,0])), torch.tensor(np.abs(y[:,0])))}')

        # Plot actual vs predicted psi for another data point
        V, E, Psi = val_set[i+1]
        V = torch.tensor([V]).to(device)
        predicted = model_Psi(V)
        x = Psi
        y = predicted.detach().numpy()[0]
        plt.subplot(133)
        plt.plot(np.abs(x[:,0]),label='Actual', color='red')
        plt.plot(np.abs(y[:,0]),label='Predicted', color='blue')
        plt.ylabel('Wavefunction')
        plt.xlabel('x')
        plt.legend()
        plt.title(f'Error = {ComplexMSELoss()(torch.Tensor(np.abs(x[:,0])), torch.tensor(np.abs(y[:,0])))}')

        # Loss vs Epoch
        plt.subplot(131)
        plt.semilogy(losses_Psi[:])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        # Calculate predicted E
        predicted = model_E(V.float()).float()
        x = E
        y = predicted.detach().numpy()[0]
        plt.figure(1)

        # Plot actual vs predicted E
        plt.subplot(122)
        x = np.abs(x)**2
        y = np.abs(y)**2
        plt.plot(x,label='Actual', color='red')
        plt.plot(y,label='Predicted', color='blue')
        plt.title(f'Error = {torch.nn.MSELoss()(torch.Tensor(np.abs(x)), torch.tensor(np.abs(y)))}')
        plt.ylabel('Energy')
        plt.xlabel('Solution Number')
        plt.grid()
        plt.legend()

        # Loss vs Epoch
        plt.subplot(121)
        plt.plot(losses_E)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.tight_layout()
        plt.show()