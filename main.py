import random
import sys
from config import Config
from Net import TreeNet
from Encoder import SqlEncoder, PlanEncoder, ValueExtractor
from Net import SPINN
from Trainer import Trainer
from Connector import PGConnector
from sklearn.model_selection import train_test_split
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import json
import torch

config = Config()
pgrunner = PGConnector(config.database, config.username, config.password, config.pghost, config.pgport)   
data = []
labels = []
plan_times = []
sqlInform = {}
tables = []

def is_json_file_empty(file_path):
    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
            if data:
                return False
            else:
                return True
        except json.JSONDecodeError:
            return True


with open(config.query_path, 'r') as f:
    for line in f:
        data_line = line.strip("\n")
        data.append(data_line)

count = 1
if is_json_file_empty(config.query_json_path):
    for sql in data:
        print("执行第{}条sql".format(count))
        plan_json = pgrunner.getPGPlan(sql)
        exe_time = pgrunner.getPGPlan(sql)['Plan']['Actual Total Time']
        plan_time = pgrunner.getPGPlan(sql)['Planning Time']
        sqlInform[sql] = (plan_json, exe_time, plan_time)
        labels.append(exe_time)
        count = count + 1
    with open(config.query_json_path, "w") as json_file:
        json.dump(sqlInform, json_file)
        print("成功写入json！")
else:
    with open(config.query_json_path, "r") as json_file:
        sqlInform = json.load(json_file)
        for sql in data:
            labels.append(sqlInform[sql][1])


train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=config.split_rate, random_state=42)       
table2index, tables = pgrunner.getAllTables()

value_extractor = ValueExtractor()
tree_builder = PlanEncoder(table2index)
sql2vec = SqlEncoder(table2index)
value_network = SPINN(input_size=config.input_size, hidden_size=config.hidden_size, table_num = config.max_table_num,sql_size = config.max_table_num*config.max_table_num+config.max_column).to(config.device)
for name, param in value_network.named_parameters():
    from torch.nn import init
    if len(param.shape)==2:
        init.xavier_normal(param)
    else:
        init.uniform(param)
net = TreeNet(tree_builder= tree_builder,value_network = value_network)
trainer = Trainer(net, sql2vec,value_extractor, sqlInform, table2index)

count = 0
x = []
losses = []
min_loss = 999
# for epoch in range(100):
#     for query in train_x:
#          loss = trainer.train(query, epoch)
#     loss = 0
#     for query in test_x:
#          loss = loss + trainer.validate(query)
#     x.append(count)
#     count = count+1
#     losses.append(loss.item())
#     if loss.item() < min_loss:
#         trainer.save()
#         min_loss = loss.item()
#     print("epoch:{}, loss:{}".format(epoch, loss))


trainer.model.load_state_dict(torch.load(config.model_path))
trainer.model.eval()
tuples = []
x_ = []
times = []
count = 0
for query in train_x:
    x_.append(count)
    count = count + 1 
    time = trainer.get_time(query)
    times.append(sqlInform[query][1])
    tuples.append((query, sqlInform[query][1], time))
    print("real time:{} predicted time:{}".format(sqlInform[query][1], time))
def by_real_time(t):
    return t[1]
sorted_tuples = sorted(tuples, key=by_real_time)
y1 = []
y2 = []
for tuple in sorted_tuples:
    y1.append(tuple[1])
    y2.append(tuple[2])

plt.scatter(x_, y1)
plt.scatter(x_, y2)

# plt.plot(x,losses)
plt.savefig("img/1.png")
plt.cla()
plt.hist(times,bins=10)
plt.savefig("img/2.png")
plt.close()
