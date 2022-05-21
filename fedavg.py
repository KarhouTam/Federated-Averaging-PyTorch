from sys import path

path.append("../")

import torch
from fedlab.core.client import ClientTrainer
from fedlab.utils.serialization import SerializationTool
from tqdm import trange
from data import get_dataloader
from copy import deepcopy
from torch import optim
from utils import evaluate


class FedAvgTrainer(ClientTrainer):
    def __init__(
        self, client_id, global_model, dataset, batch_size, lr, criterion, epochs, cuda,
    ):
        super().__init__(deepcopy(global_model), cuda and torch.cuda.is_available())
        self.device = next(iter(self.model.parameters())).device
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.criterion = criterion
        self.lr = lr
        self.batch_size = batch_size
        self.dataset = dataset
        self.trainloader, self.valloader = get_dataloader(
            client_id, dataset, batch_size
        )
        self.id = client_id
        self.iter_trainloader = iter(self.trainloader)
        self.iter_valloader = iter(self.valloader)

    def train(self, global_model_parameters):
        SerializationTool.deserialize_model(self.model, global_model_parameters)

        return self._train(self.model, self.optimizer, self.epochs)

    def eval(self, global_model_parameters):
        # using client local model's replica for evaluating
        model_4_eval = deepcopy(self.model)
        optimizer = optim.SGD(model_4_eval.parameters(), lr=self.lr)
        SerializationTool.deserialize_model(model_4_eval, global_model_parameters)
        # evaluate global FedAvg performance
        loss_g, acc_g = evaluate(model_4_eval, self.valloader, self.criterion, self.device)
        # localization
        self._train(model_4_eval, optimizer, 10)
        # evaluate localized FedAvg performance
        loss_l, acc_l = evaluate(model_4_eval, self.valloader, self.criterion, self.device)

        return loss_g, acc_g, loss_l, acc_l

    def _train(self, model, optimizer, epochs):
        model.train()
        for _ in trange(epochs, desc="client [{}]".format(self.id)):
            x, y = self.get_data_batch(train=True)
            logit = model(x)
            loss = self.criterion(logit, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        weight = torch.tensor(len(self.trainloader.dataset), dtype=torch.float)
        return weight, SerializationTool.serialize_model(model)

    def get_data_batch(self, train: bool):
        if train:
            try:
                data, targets = next(self.iter_trainloader)
            except StopIteration:
                self.iter_trainloader = iter(self.trainloader)
                data, targets = next(self.iter_trainloader)
        else:
            try:
                data, targets = next(self.iter_valloader)
            except StopIteration:
                self.iter_valloader = iter(self.valloader)
                data, targets = next(self.iter_valloader)

        return data.to(self.device), targets.to(self.device)
