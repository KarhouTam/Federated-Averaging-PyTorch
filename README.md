# Federated-Averaging-PyTorch 

This repo is the implementation of [FedAvg](http://arxiv.org/abs/1602.05629)

For simulating Non-I.I.D scenario, the dataset is split by labels and each client has only **two** classes of data.

# Requirements

path~=16.4.0

torch~=1.10.2

numpy~=1.21.2

fedlab~=1.1.4

torchvision~=0.11.3

tqdm~=4.62.3

```python
pip install -r requirements.txt
```



# Run the experiment

It’s so simple.

MNIST, CIFAR-10, [Synthetic](https://arxiv.org/abs/1812.06127) are supported. Have fun!🤪

```python
python main.py
```



## Hyperparameters

`--comms_round`: Num of communication rounds. Default: `40`

`--dataset`: Name of experiment dataset. Default: `mnist`

`--client_num_per_round`: Num of clients that participating training at each communication round. Default: `5`

`--test_round`: Num of round for final evaluation. Default: `1`

`--local_lr`: Learning rate for client local model updating. Default: `0.05`

`--batch_size`: Batch size of client local dataset.

`--global_lr`: Learning rate for server model updating. Default: `1.0`

`--cuda`: `True` for using GPU. Default: `True`

`--epochs`: Num of local epochs in client training. Default: `5`

`--model`: Structure of model. Must be `mlp` or `cnn`. Default: `cnn`




# Result

| Algorithm | Global Loss | Localized Loss | Global Acc | Localized Acc |
| --------- | ----------- | -------------- | ---------- | ------------- |
| FedAvg    | `1.6071`    | `0.2009`       | `89.80%`   | `98.80%`      |

Localization means the model additionally train for 10 local epochs at the final evaluation phase, which is for adapting client’s local dataset.

