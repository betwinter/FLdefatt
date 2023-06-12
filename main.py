import argparse
import json
import random
import shuju
import torch.nn as nn
from client import *
from server import *
from maliciousClient import  *
import numpy as np

def EuclideanDistances(a, b):
    pdist = nn.PairwiseDistance(p=2)
    output = pdist(a, b)
    return output


if __name__ == '__main__':


    conf = { "model_name": "resnet18",
    "no_models": 10,
    "type": "cifar",
    "global_epochs": 20,
    "local_epochs": 2,
    "k":6,
    "batch_size": 32,
    "lr": 0.001,
    "momentum": 0.0001,
    "lambda": 0.1}

    train_datasets, eval_datasets = shuju.Dataset.get_dataset("data/cifar-10-pythoN", conf["type"])

    server = Server(conf, eval_datasets)
    clients = []
    malicious = []
    clients2 = []


    for c in range(conf["no_models"]-1):
        clients.append(Client(conf, server.global_model, train_datasets, c))
    clients.append(malClient(conf, server.global_model, train_datasets, 10))

    print("\n\n")

    # 全局模型训练
    for e in range(conf["global_epochs"]):
        print("Global Epoch %d" % e)
        candidates = random.sample(clients, conf["k"])
        print("select clients is: ")
        for c in candidates:
            print(c.client_id)

        # 权重累计
        weight_accumulator = {}
        median = {}
        median1 = {}
        median2 = {}
        mal = {}
        high = {}
        low = {}
        mall = 0
        alllow = {}
        allhigh = {}
        all = {}
        final = {}
        change = {}
        most = {}
        i = 0
        aa = 0
        bb = 0
        most1 = {}
        change1 = {}
        most2 = {}

        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)


        for c in candidates:
            diff = c.local_train(server.global_model)
            all[i] = diff
            i = i+1
            for name, params in server.global_model.state_dict().items():
                if bool(1 - bool(name in mal.keys())):
                    median[name] = diff[name]
                    high[name ] = diff[name]
                    low[name] = diff[name]
                else:
                    median[name] += diff[name]
                if np.sum(EuclideanDistances(high[name],diff[name]).numpy!=0)>np.sum(EuclideanDistances(high[name],diff[name]).numpy==0):
                    high[name] = diff[name]
                if np.sum(EuclideanDistances(low[name],diff[name]).numpy!=0)< np.sum(EuclideanDistances(low[name],diff[name]).numpy==0):
                    low[name] = diff[name]
            for name, params in server.global_model.state_dict().items():
                if name in low.keys() & name in high.keys() & low[name].size() == high[name].size():
                    median1[name] = median[name]/6+(EuclideanDistances(low[name],high[name]))/5
                    median2[name] = median[name]/6-(EuclideanDistances(low[name],high[name]))/5
                elif name in low.keys():
                    median1[name] = median[name]
                    median2[name] = median[name]
                elif name in high.keys():
                    median1[name] = median[name]
                    median2[name] = median[name]

                    mal[name] = median[name]*0.75


            if c.client_id in range(7,10):
                for name, params in server.global_model.state_dict().items():
                    if np.sum(torch.lt(diff[name], torch.tensor(median1[name])).numpy != 0) > np.sum(
                            torch.lt(diff[name], torch.tensor(median1[name])).numpy == 0) | np.sum(
                            torch.gt(diff[name], torch.tensor(median2[name])).numpy != 0) > np.sum(
                            torch.gt(diff[name], torch.tensor(median2[name])).numpy == 0):
                        weight_accumulator[name].add_(diff[name])
                       
            else:
                for name, params in server.global_model.state_dict().items():
                    if np.sum(torch.lt(diff[name],torch.tensor(median1[name])).numpy!=0)>np.sum(torch.lt(diff[name],torch.tensor(median1[name])).numpy==0) | np.sum(torch.gt(diff[name],torch.tensor(median2[name])).numpy!=0)>np.sum(torch.gt(diff[name],torch.tensor(median2[name])).numpy==0):

                        weight_accumulator[name].add_(diff[name])
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])
        for name, params in server.global_model.state_dict().items():
            for i in range(len(all)):
                if i == 0:
                    most[name] = all[0][name] - all[1][name]
                    most1[name] = all[0][name] + all[1][name]
                    most2[name] = all[0][name] + all[1][name]
                    change1[name] = all[0][name] - all[1][name]
                for j in range(len(all)):
                    if j!=i:
                       change[name] = all[i][name] - all[j][name]
                       if np.sum(EuclideanDistances(change1[name], change[name]).numpy != 0) > np.sum(EuclideanDistances(change1[name], change[name]).numpy == 0):
                            most[name] = all[i][name] + all[j][name]
                            change1[name] = change[name]
                       if np.sum(EuclideanDistances(most1[name], most[name]).numpy != 0) > np.sum(
                           EuclideanDistances(most1[name], most[name]).numpy == 0):
                           most1[name] = most[name]
                           aa = i
                           bb = j
                       if np.sum(EuclideanDistances(most1[name], most[name]).numpy != 0) < np.sum(
                           EuclideanDistances(most1[name], most[name]).numpy == 0):
                           most2[name] = most[name]
                           aa = i
                           bb = j

            weight_accumulator[name].add_(all[aa][name] + all[bb][name]+(1/5*most1[name]+most2[name]).long())
            '''weight_accumulator[name].add_(all[aa][name] + all[bb][name])'''





        server.model_aggregate(weight_accumulator)


        acc, loss = server.model_eval()

        print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))

    for c in range(conf["no_models"] - 3):
        clients2.append(Client(conf, server.global_model, train_datasets, c))

    for e in range(conf["global_epochs"]+10):
        print("Epoch %d" % e)
        for i in range(conf["no_models"]-3):
            c = clients2[i]
            print("select clients is: ")
            print(i)
            weight_accumulator = {}
            for name, params in server.global_model.state_dict().items():
                # 生成一个和参数矩阵大小相同的0矩阵
                weight_accumulator[name] = torch.zeros_like(params)

            diff = c.local_train(server.global_model)
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name] )
            server.model_aggregate(weight_accumulator)
            acc, loss = server.model_eval()
            print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))
