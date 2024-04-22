from config import Config
from math import e 
from Connector import PGConnector
import torch
import torch.nn.functional as F
import time

config = Config()
pgrunner = PGConnector(config.database, config.username, config.password, config.pghost, config.pgport)   
def formatFloat(t):
    try:
        return " ".join(["{:.4f}".format(x) for x in t])
    except:
        return " ".join(["{:.4f}".format(x) for x in [t]])


class Timer:
    def __init__(self,):
        from time import time
        self.timer = time
        self.startTime = {}
    def reset(self,s):
        self.startTime[s] = self.timer()
    def record(self,s):
        return self.timer()-self.startTime[s]
timer = Timer()

class Trainer:
    def __init__(self,model,sql2vec,value_extractor, sqlInform, table2index):
        self.model = model #Net.TreeNet
        self.sql2vec = sql2vec#
        self.value_extractor = value_extractor
        self.pg_planningtime_list = []
        self.pg_runningtime_list = [] #default pg running time
        self.sqlInform = sqlInform
        self.table2index = table2index

    def train(self,sql):

        plan_json = self.sqlInform[sql][0]
        exe_time = self.sqlInform[sql][1]
        
        sql_vec = self.sql2vec.to_vec(sql)
        sql_feature = self.model.value_network.sql_feature(sql_vec)
        plan_feature = self.model.tree_builder.plan_to_feature_tree(plan_json)
        target_value = self.value_extractor.encode(exe_time)
        loss = self.model.train(plan_json = plan_json,sql_vec = sql_vec,target_value=target_value)
        return loss
    
    def validate(self, sql):
        plan_json = self.sqlInform[sql][0]
        exe_time = self.sqlInform[sql][1]

        sql_vec = self.sql2vec.to_vec(sql)
        sql_feature = self.model.value_network.sql_feature(sql_vec)
        plan_feature = self.model.tree_builder.plan_to_feature_tree(plan_json)
        value = self.model.plan_to_value(plan_feature=plan_feature,sql_feature = sql_feature)
        target_value = self.value_extractor.encode(exe_time)
        loss_value = F.mse_loss(value, target_value)
        return loss_value

