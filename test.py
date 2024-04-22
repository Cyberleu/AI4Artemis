import random
import sys
from config import Config
from Net import TreeNet
from Encoder import SqlEncoder, PlanEncoder, ValueExtractor
from Net import SPINN
from Trainer import Trainer
from Connector import PGConnector
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# config = Config()
# pgrunner = PGConnector(config.database, config.username, config.password, config.pghost, config.pgport)   
# data = []
# labels = []
# plan_times = []
# sqlInform = {}
# with open(config.query_path, 'r') as f:
#     for line in f:
#         data_line = line.strip("\n")
#         data.append(data_line)

# for sql in data:
#         plan_json = pgrunner.getPGPlan(sql)
#         exe_time = pgrunner.getPGPlan(sql)['Plan']['Actual Total Time']
#         plan_time = pgrunner.getPGPlan(sql)['Planning Time']
#         sqlInform[sql] = (plan_json, exe_time, plan_time)
#         labels.append(exe_time)

# train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=config.split_rate, random_state=42)       
# pgrunner.getAllTables()
# k = (torch.rand((1,2)),np.asarray([1,2,3]))

x = [1,2,3]
losses = [3,2,1]
plt.figure(figsize=(10, 5), dpi=400)
plt.plot(x,losses)
plt.savefig("img/1.png")
plt.close()