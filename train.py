import os
import wandb
from time import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from qecsim.models.toric import ToricCode
from qecsim.models.generic import BitFlipErrorModel, DepolarizingErrorModel

from dataset import generate_dataset
from network import Net


def train(model, dataloader, device, save_path,
          lr=0.001, epochs=5, weight=[1.0, 3.0, 3.0, 3.0]):
    model = model.to(device)
    if os.path.exists(save_path):
        print("loading exist weights from "+save_path)
        model.load_state_dict(torch.load(save_path))

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight).to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                     factor=0.5, patience=10, cooldown=5, min_lr=1e-6)

    start = time()
    print("Training started:")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        epoch_loss = 0.0

        for i, (_, err_sydns, targets) in enumerate(dataloader):
            err_sydns = err_sydns.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(err_sydns.to(torch.float32))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()*len(err_sydns)
            running_loss += loss.item()
            if i % 29 == 0:
                wandb.log(
                    {"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                scheduler.step(running_loss)
                running_loss = 0.0
                print(loss.item())

        torch.save(model.state_dict(), save_path)
        epoch_loss = epoch_loss / len(dataloader.dataset)
        wandb.log({"avg loss": epoch_loss})
        now = time() - start
        print("average loss over epoch {}: {:.4f}, cost time: {:.0f}m {:.0f}s".format(
            epoch + 1, epoch_loss, now // 60, now % 60))
    return


if __name__ == "__main__":
    codes = [ToricCode(*size) for size in [(7, 7), (9, 9), (11, 11), (13, 13)]]
    my_error_model = BitFlipErrorModel()
    my_error_probabilities = [0.01, 0.05, 0.09]
    # my_error_model = DepolarizingErrorModel()
    # my_error_probabilities = [0.03, 0.07, 0.11, 0.15]

    assert torch.cuda.is_available()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_data_size = 10**6
    batch_size = 256
    learning_rate = 0.01
    epochs = 3
    weight = [1.0] + [1.0]*(1 if my_error_model.label == "Bit-flip" else 3)
    model = Net(6, 12, num_class=len(weight))

    for my_code in codes:
        folder_path = os.path.join(my_code.label, my_error_model.label)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for my_error_probability in my_error_probabilities:
            wandb.init(
                project="U-Net for QEC",
                entity="chirtee",
                name=my_code.label + "_" + my_error_model.label +
                "_" + str(my_error_probability),
                reinit=True,
                tags=["5"],
                config={
                    "code": my_code.label,
                    "error_modle": my_error_model.label,
                    "error_probability": my_error_probability,
                    "train_data_size": train_data_size,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "epochs": epochs,
                    "weight": weight,
                    "n_data_aug": 1
                }
            )
            file_name = "weights_" + str(my_error_probability) + ".pth"

            dataset = generate_dataset(my_code, my_error_model, my_error_probability,
                                       data_size=train_data_size, n_data_aug=1, mode="train")
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
            train(model, dataloader, device, save_path=os.path.join(folder_path, file_name),
                  lr=learning_rate, epochs=epochs, weight=weight)
            wandb.finish()
