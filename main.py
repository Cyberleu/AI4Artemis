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
for epoch in range(100):
    for query in train_x:
         loss = trainer.train(query)
    loss = 0
    for query in test_x:
         loss = loss + trainer.validate(query)
    x.append(count)
    count = count+1
    losses.append(loss.item())
    print("epoch:{}, loss:{}".format(epoch, loss))

plt.plot(x,losses)
plt.savefig("img/1.png")
plt.close()
