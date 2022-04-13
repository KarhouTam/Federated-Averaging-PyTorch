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
        self, global_model, lr, criterion, epochs, cuda,
    ):
        super().__init__(deepcopy(global_model), cuda and torch.cuda.is_available())
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.criterion = criterion
        self.lr = lr

    def train(self, client_id, global_model_parameters, dataset, batch_size):
        trainloader, _ = get_dataloader(client_id, dataset, batch_size)
        SerializationTool.deserialize_model(self.model, global_model_parameters)

        return self._train(
            self.model, self.optimizer, trainloader, self.epochs, client_id
        )

    def eval(self, client_id, global_model_parameters, dataset, batch_size):
        trainloader, testloader = get_dataloader(client_id, dataset, batch_size)
        # using client local model's replica for evaluating
        model_4_eval = deepcopy(self.model)
        optimizer = optim.SGD(model_4_eval.parameters(), lr=self.lr)
        SerializationTool.deserialize_model(model_4_eval, global_model_parameters)
        # evaluate global FedAvg performance
        loss_g, acc_g = evaluate(model_4_eval, testloader, self.criterion, self.gpu)
        # localization
        self._train(model_4_eval, optimizer, trainloader, 10, client_id)
        # evaluate localized FedAvg performance
        loss_l, acc_l = evaluate(model_4_eval, testloader, self.criterion, self.gpu)

        return loss_g, acc_g, loss_l, acc_l

    def _train(self, model, optimizer, trainloader, epochs, client_id):
        model.train()
        for _ in trange(epochs, desc="client [{}]".format(client_id)):
            for x, y in trainloader:
                if self.cuda:
                    x, y = x.to(self.gpu), y.to(self.gpu)
                logit = model(x)
                loss = self.criterion(logit, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        weight = torch.tensor(len(trainloader.dataset), dtype=torch.float)
        return weight, SerializationTool.serialize_model(model)
