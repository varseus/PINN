import torch
from torch.utils.data import DataLoader, random_split
from model import FullyConnectedNetwork_E, FullyConnectedNetwork_Psi, init_weights_E, init_weights_Psi
from data_loader import Dataset_Random_Potentials
from loss_function import ComplexMSELoss, upper_bound_loss
from datetime import datetime
import pickle


def train_1_epoch(dataloader, target_type, model, loss_fn, optimizer):
    """ Training a model on a dataset for 1 epoch

    Args:
        dataloader: iterable with batches of data 
        model: PyTorch model to train
        loss_fn: loss function
        optimizer: optizer

    Returns:
        float: accuracy
        float: total_loss
    """
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0

    for batch, (V, E, Psi) in enumerate(dataloader):
        target = E.float() if target_type=='E' else Psi.cfloat()
        V, target = V.float().to(device), target.to(device)

        # Compute prediction error
        predicted = model(V)
        loss = loss_fn(predicted, target)
        
        # Backpropagation
        if target_type=='E' and loss >= 0.1:
            continue
        total_loss += loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Print results
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(V)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # Save results
    torch.save(model.state_dict(), f'{model_name}_{target_type}')
    print(f'Training Loss: {total_loss}')

    return float(total_loss)

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
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for V, E, Psi in dataloader:
            target = E if target_type=='E' else Psi
            V, target = V.float().to(device), target.to(device)
            pred = model(V)
            total_loss += loss_fn(pred, target).item()
    print(f'Test Loss: {total_loss} \n')



if __name__ == "__main__":
    # Get cpu, gpu or mps device
    device = ("cuda" if torch.cuda.is_available() else "cpu")#"mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using {device} device")
    model_name = 'models/model_' + datetime.today().strftime('%Y-%m-%d')

    # Load Data
    batch_size = 64
    loss_fn = upper_bound_loss # MSE is too sensitive to outliers
    loss_fn_complex = ComplexMSELoss()
    training_data = Dataset_Random_Potentials(load='data/randomly_generated_dataset.pkl')
    train_set, val_set = random_split(training_data, [70000, 10000], generator=torch.Generator().manual_seed(1))
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # Run for E
    epochs = 25
    model_E = FullyConnectedNetwork_E().to(device)
    model_E.apply(init_weights_E)
    optimizer_E = torch.optim.Adam(model_E.parameters(), lr=1e-5)
    scheduler_E = torch.optim.lr_scheduler.StepLR(optimizer_E, step_size=1, gamma=0.95)
    e_loss = []
    for epoch in range(epochs):
        print(f'Epoch: {epoch} | LR: {optimizer_E.param_groups[0]["lr"]}')
        e_loss.append(train_1_epoch(dataloader=train_dataloader, target_type='E', model=model_E, loss_fn=loss_fn, optimizer=optimizer_E))
        e_loss.append(test_1_epoch(dataloader=test_dataloader, target_type='E', model=model_E, loss_fn=loss_fn))

        # Save loss vs epoch
        with open('models/e_loss.pkl', 'wb') as f:
            pickle.dump(e_loss, f) # loss as a function of epoch

        scheduler_E.step()
    
    # Run for Psi
    epochs = 150
    model_Psi = FullyConnectedNetwork_Psi().to(device)
    model_Psi.apply(init_weights_Psi)
    optimizer_Psi = torch.optim.Adam(model_Psi.parameters(), lr=1)
    scheduler_Psi = torch.optim.lr_scheduler.StepLR(optimizer_Psi, step_size=4, gamma=0.5)
    psi_loss = []
    for epoch in range(epochs):
        if epoch in [36, 75, 110]: # occasionally reset the learning rate so it does not decay too much, and does not get stuck at local minima
            optimizer_Psi = torch.optim.Adam(model_Psi.parameters(), lr=.1)
            scheduler_Psi = torch.optim.lr_scheduler.StepLR(optimizer_Psi, step_size=4, gamma=0.5)

        print(f'Epoch: {epoch} | LR: {optimizer_Psi.param_groups[0]["lr"]}')
        psi_loss.append(train_1_epoch(dataloader=train_dataloader, target_type='Psi', model=model_Psi, loss_fn=loss_fn_complex, optimizer=optimizer_Psi))
        psi_loss.append(test_1_epoch(dataloader=test_dataloader, target_type='Psi', model=model_Psi, loss_fn=loss_fn_complex))

        # Save loss vs epoch
        with open('models/psi_loss.pkl', 'wb') as f:
            pickle.dump(psi_loss, f) # loss as a function of epoch

        scheduler_Psi.step()