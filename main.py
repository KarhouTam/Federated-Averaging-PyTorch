import torch
from random import sample
from torch.nn import CrossEntropyLoss
from fedavg import FedAvgTrainer
from utils import get_args, get_model
from argparse import ArgumentParser
from tqdm import trange
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.aggregator import Aggregators
from os import listdir

# ================== Can not remove these modules ===================
# these modules are imported for pickles.load deserializing properly.
from data.cifar import CIFARDataset
from data.mnist import MNISTDataset
from data.synthetic import SyntheticDataset
# ===================================================================
if __name__ == "__main__":
    parser = ArgumentParser()
    args = get_args(parser)

    global_model = get_model((args.model, args.dataset))
    criterion = CrossEntropyLoss()
    trainer = FedAvgTrainer(
        global_model=global_model,
        lr=args.local_lr,
        criterion=criterion,
        epochs=args.epochs,
        cuda=args.cuda,
    )
    aggregator = Aggregators.fedavg_aggregate
    client_num_in_total = len(listdir("data/{}/pickles".format(args.dataset)))
    client_indices = range(client_num_in_total)

    for r in trange(args.comms_round, desc="\033[1;33mtraining epoch\033[0m"):
        # select clients
        selected_clients = sample(client_indices, args.client_num_per_round)
        print(
            "\033[1;34mselected clients in round [{}]: {}\033[0m".format(
                r, selected_clients
            )
        )
        global_model_param = SerializationTool.serialize_model(global_model)
        weights_buffer = []
        params_buffer = []
        # train
        for client_id in selected_clients:
            weight, serialized_param = trainer.train(
                client_id, global_model_param, args.dataset, args.batch_size
            )
            weights_buffer.append(weight)
            params_buffer.append(serialized_param)

        # aggregate models
        with torch.no_grad():
            aggregated_param = aggregator(params_buffer, weights_buffer)
            SerializationTool.deserialize_model(global_model, aggregated_param)

    # evaluate
    avg_loss_g = 0  # global model loss
    avg_acc_g = 0  # global model accuracy
    avg_loss_l = 0  # localized model loss
    avg_acc_l = 0  # localized model accuracy
    for r in trange(args.test_round, desc="\033[1;36mevaluating epoch\033[0m"):
        selected_clients = sample(client_indices, args.client_num_per_round)
        print(
            "\033[1;34mselected clients in round [{}]: {}\033[0m".format(
                r, selected_clients
            )
        )
        global_model_param = SerializationTool.serialize_model(global_model)
        for client_id in selected_clients:
            stats = trainer.eval(
                client_id, global_model_param, args.dataset, args.batch_size
            )
            avg_loss_g += stats[0]
            avg_acc_g += stats[1]
            avg_loss_l += stats[2]
            avg_acc_l += stats[3]

    # display experiment results
    avg_loss_g /= args.client_num_per_round * args.test_round
    avg_acc_g /= args.client_num_per_round * args.test_round
    avg_loss_l /= args.client_num_per_round * args.test_round
    avg_acc_l /= args.client_num_per_round * args.test_round
    print("\033[1;32m---------------------- RESULTS ----------------------\033[0m")
    print("\033[1;33m Global FedAvg loss: {:.4f}\033[0m".format(avg_loss_g))
    print("\033[1;33m Global FedAvg accuracy: {:.2f}%\033[0m".format(avg_acc_g))
    print("\033[1;36m Localized FedAvg loss: {:.4f}\033[0m".format(avg_loss_l))
    print("\033[1;36m Localized FedAvg accuracy: {:.2f}%\033[0m".format(avg_acc_l))
