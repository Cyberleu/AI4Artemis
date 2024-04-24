import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from config import Config
import numpy as np

config = Config()
class TreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.fc_left = nn.Linear(hidden_size, 5 * hidden_size)
        self.fc_right = nn.Linear(hidden_size, 5 * hidden_size)
        self.fc_input = nn.Linear(input_size, 5 * hidden_size)
        elementwise_affine = False
        self.layer_norm_input = nn.LayerNorm(5 *hidden_size,elementwise_affine = elementwise_affine)
        self.layer_norm_left = nn.LayerNorm(5 *hidden_size,elementwise_affine = elementwise_affine)
        self.layer_norm_right = nn.LayerNorm(5 *hidden_size,elementwise_affine = elementwise_affine)
        self.layer_norm_c = nn.LayerNorm(hidden_size,elementwise_affine = elementwise_affine)
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, h_left,c_left,h_right,c_right,feature):
        lstm_in = self.layer_norm_left(self.fc_left(h_left))
        lstm_in += self.layer_norm_right(self.fc_right(h_right))
        lstm_in += self.layer_norm_input(self.fc_input(feature))
        a, i, f1, f2, o = lstm_in.chunk(5, 1)
        c = (a.tanh() * i.sigmoid() + f1.sigmoid() * c_left +
             f2.sigmoid() * c_right)
        c = self.layer_norm_c(c)
        h = o.sigmoid() * c.tanh()
        return h,c

class SPINN(nn.Module):
    def __init__(self, input_size, hidden_size, table_num, sql_size):
        super(SPINN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tree_lstm = TreeLSTM(input_size, hidden_size)
        self.sql_layer = nn.Linear(sql_size, hidden_size)
        self.layer = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size//2),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size//2,1))
        self.table_embeddings = nn.Embedding(table_num, hidden_size)
        self.relu = nn.ReLU()
    def input_feature(self,feature):
        return torch.tensor(feature,device = config.device,dtype = torch.float32).reshape(-1,self.input_size)
    def sql_feature(self,feature):
        return torch.tensor(feature,device = config.device,dtype = torch.float32).reshape(1,-1)
    def tree_node(self, h_left,c_left,h_right,c_right,feature):
        h,c =  self.tree_lstm(h_left,c_left,h_right,c_right,feature)
        return (h,c)
    def leaf(self, alias_id):
        table_embedding  = self.table_embeddings(alias_id)
        return (table_embedding, torch.zeros(table_embedding.shape,device = config.device,dtype = torch.float32))
    def logits(self, encoding,sql_feature,prt=False):
        sql_hidden = self.relu(self.sql_layer(sql_feature))
        out_encoding = torch.cat([encoding,sql_hidden],dim = 1)
        out = self.layer(out_encoding)
        return out
    
    def zero_hc(self,input_dim = 1):
        return (torch.zeros(input_dim,self.hidden_size,device = config.device),torch.zeros(input_dim,self.hidden_size,device = config.device))
        
        
class TreeNet:
    def __init__(self, tree_builder, value_network):
        self.value_network = value_network
        self.tree_builder = tree_builder
        self.optimizer = optim.Adam(self.value_network.parameters(), lr = 3e-6, betas = (0.9, 0.999))
    def plan_to_value(self, plan_feature, sql_feature):
        def recursive(tree_feature):
            if isinstance(tree_feature[1],tuple):
                feature = tree_feature[0]
                h_left,c_left = recursive(tree_feature=tree_feature[1])
                h_right,c_right = recursive(tree_feature=tree_feature[2])
                return self.value_network.tree_node(h_left,c_left,h_right,c_right,feature)
            else:
                feature = tree_feature[0]
                h_left,c_left = self.value_network.leaf(tree_feature[1])
                h_right,c_right = self.value_network.zero_hc()
                return self.value_network.tree_node(h_left,c_left,h_right,c_right,feature)
        plan_feature = recursive(tree_feature=plan_feature)
        value = self.value_network.logits(plan_feature[0],sql_feature)
        return value
    def loss(self, value, target_value, optimize = True):
        loss_value = F.mse_loss(value, target_value)
        if not optimize:
            return loss_value.item()
        self.optimizer.zero_grad()
        loss_value.backward()
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-2, 2)
        self.optimizer.step()
        return loss_value.item()
    def train(self, plan_json, sql_vec, target_value):
        plan_feature = self.tree_builder.plan_to_feature_tree(plan_json)
        sql_feature = self.value_network.sql_feature(sql_vec)
        value = self.plan_to_value(plan_feature=plan_feature,sql_feature = sql_feature)
        loss_value = self.loss(value, target_value)
        return loss_value
