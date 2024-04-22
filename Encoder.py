from psqlparse import parse_dict
import numpy as np
from Connector import PGConnector
from config import Config
import torch
class Expr:
    def __init__(self, expr,list_kind = 0):
        self.expr = expr
        self.list_kind = list_kind
        self.isInt = False
        self.isFloat = False
        self.val = 0
    def isCol(self,):
        return isinstance(self.expr, dict) and "ColumnRef" in self.expr

    def getValue(self, value_expr):
        if "A_Const" in value_expr:
            value = value_expr["A_Const"]["val"]
            if "String" in value:
                return "'" + value["String"]["str"].replace("'","''")+"\'"
            elif "Integer" in value:
                self.isInt = True
                self.val = value["Integer"]["ival"]
                return str(value["Integer"]["ival"])
            elif "Float" in value:
                self.isFloat = True
                self.val = value["Float"]["str"]
                return str(self.val)

        elif "TypeCast" in value_expr:
            if len(value_expr["TypeCast"]['typeName']['TypeName']['names'])==1:
                return value_expr["TypeCast"]['typeName']['TypeName']['names'][0]['String']['str']+" '"+value_expr["TypeCast"]['arg']['A_Const']['val']['String']['str']+"'"
            else:
                if value_expr["TypeCast"]['typeName']['TypeName']['typmods'][0]['A_Const']['val']['Integer']['ival']==2:
                    return value_expr["TypeCast"]['typeName']['TypeName']['names'][1]['String']['str']+" '"+value_expr["TypeCast"]['arg']['A_Const']['val']['String']['str']+ "' month"
                else:
                    return value_expr["TypeCast"]['typeName']['TypeName']['names'][1]['String']['str']+" '"+value_expr["TypeCast"]['arg']['A_Const']['val']['String']['str']+ "' year"
        else:
            print(value_expr.keys())
            raise "unknown Value in Expr"

    def getTableName(self,):
        return self.expr["ColumnRef"]["fields"][0]["String"]["str"]

    def getColumnName(self,):
        return self.expr["ColumnRef"]["fields"][1]["String"]["str"]

    def __str__(self,):
        if self.isCol():
            return self.getTableName()+"."+self.getColumnName()
        elif isinstance(self.expr, dict) and "A_Const" in self.expr:
            return self.getValue(self.expr)
        elif isinstance(self.expr, dict) and "TypeCast" in self.expr:
            return self.getValue(self.expr)
        elif isinstance(self.expr, list):
            if self.list_kind == 6:
                return "("+",\n".join([self.getValue(x) for x in self.expr])+")"
            elif self.list_kind == 10:
                return " AND ".join([self.getValue(x) for x in self.expr])
            else:
                raise "list kind error"


class Comparison:
    def __init__(self, comparison):
        self.comparison = comparison
        self.column_list = []
        if "A_Expr" in self.comparison:
            self.lexpr = Expr(comparison["A_Expr"]["lexpr"])
            self.column = str(self.lexpr)
            self.kind = comparison["A_Expr"]["kind"]
            if not "A_Expr" in comparison["A_Expr"]["rexpr"]:
                self.rexpr = Expr(comparison["A_Expr"]["rexpr"],self.kind)
            else:
                self.rexpr = Comparison(comparison["A_Expr"]["rexpr"])

            self.aliasname_list = []

            if self.lexpr.isCol():
                self.aliasname_list.append(self.lexpr.getTableName())
                self.column_list.append(self.lexpr.getColumnName())

            if self.rexpr.isCol():
                self.aliasname_list.append(self.rexpr.getTableName())
                self.column_list.append(self.rexpr.getColumnName())

            self.comp_kind = 0
        elif "NullTest" in self.comparison:
            self.lexpr = Expr(comparison["NullTest"]["arg"])
            self.column = str(self.lexpr)
            self.kind = comparison["NullTest"]["nulltesttype"]

            self.aliasname_list = []

            if self.lexpr.isCol():
                self.aliasname_list.append(self.lexpr.getTableName())
                self.column_list.append(self.lexpr.getColumnName())
            self.comp_kind = 1
        else:
            #             "boolop"
            self.kind = comparison["BoolExpr"]["boolop"]
            self.comp_list = [Comparison(x)
                              for x in comparison["BoolExpr"]["args"]]
            self.aliasname_list = []
            for comp in self.comp_list:
                if comp.lexpr.isCol():
                    self.aliasname_list.append(comp.lexpr.getTableName())
                    self.lexpr = comp.lexpr
                    self.column = str(self.lexpr)
                    self.column_list.append(comp.lexpr.getColumnName())
                    break
            self.comp_kind = 2
    def isCol(self,):
        return False
    def __str__(self,):

        if self.comp_kind == 0:
            Op = ""
            if self.kind == 0:
                Op = self.comparison["A_Expr"]["name"][0]["String"]["str"]
            elif self.kind == 7:
                if self.comparison["A_Expr"]["name"][0]["String"]["str"]=="!~~":
                    Op = "not like"
                else:
                    Op = "like"
            elif self.kind == 8:
                if self.comparison["A_Expr"]["name"][0]["String"]["str"]=="~~*":
                    Op = "ilike"
                else:
                    raise
            elif self.kind == 6:
                Op = "IN"
            elif self.kind == 10:
                Op = "BETWEEN"
            else:
                import json
                print(json.dumps(self.comparison, sort_keys=True, indent=4))
                raise "Operation ERROR"
            return str(self.lexpr)+" "+Op+" "+ str(self.rexpr)
        elif self.comp_kind == 1:
            if self.kind == 1:
                return str(self.lexpr)+" IS NOT NULL"
            else:
                return str(self.lexpr)+" IS NULL"
        else:
            res = ""
            for comp in self.comp_list:
                if res == "":
                    res += "( "+str(comp)
                else:
                    if self.kind == 1:
                        res += " OR "
                    else:
                        res += " AND "
                    res += str(comp)
            res += ")"
            return res
    def get_join_tables(self):
        return self.lexpr.getTableName(), self.rexpr.getTableName()


cfg = Config()
pgc = PGConnector(cfg.database, cfg.username, cfg.password, cfg.pghost, cfg.pgport)   
JOIN_TYPES = ["Nested Loop", "Hash Join", "Merge Join"]
LEAF_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Index Scan"]
ALL_TYPES = JOIN_TYPES + LEAF_TYPES
def normalize(cost):
    return int(np.log(2+cost)/np.log(cfg.max_time_out)*200)/200.
class SqlEncoder:
    def __init__(self, table2index):
        self.column2index = {}
        self.table2index = table2index
    def getColumnIndex(self, column):
        if column not in self.column2index.keys():
            self.column2index[column] = len(self.column2index)
        return self.column2index[column]
    def to_vec(self, sql):
        table_num = cfg.max_table_num
        parse_result = parse_dict(sql)[0]["SelectStmt"]
        self.from_tables = [x["RangeVar"] for x in parse_result["fromClause"]]
        self.join_matrix = np.zeros((table_num, table_num), dtype = np.float)
        if "BoolExpr" in parse_result["whereClause"]:
            self.comparision_list = [Comparison(x) for x in parse_result["whereClause"]["BoolExpr"]["args"]]
        else :
            self.comparision_list = [Comparison(parse_result["whereClause"])]
        self.predicates_selectivity = np.asarray([0]*cfg.max_column, dtype = float)
        for table in self.from_tables:
            table["relname"]
        for comparision in self.comparision_list:
            if len(comparision.column_list) == 2:
                left_table, right_table = comparision.get_join_tables()
                idx1 = self.table2index[left_table]
                idx2 = self.table2index[right_table]
                self.join_matrix[idx1][idx2] = 1
                self.join_matrix[idx2][idx1] = 1
            else:
                table = comparision.aliasname_list[0]
                col_index = self.getColumnIndex(comparision.column)
                selectivity = pgc.getPGSelectivity(table, str(comparision))
                self.predicates_selectivity[col_index] = self.predicates_selectivity[col_index] + selectivity
        return np.concatenate((self.join_matrix.flatten(), self.predicates_selectivity))

class PlanEncoder:
    def __init__(self, table2index):
        self.table2index = table2index
    def to_feature_cost(self, plan):
        return [normalize(plan["Total Cost"]), normalize(plan["Plan Rows"])]
    def to_table_id(self, plan):
        return np.asarray([self.table2index[plan["Relation Name"]]])
    def to_feature_join(self, plan):
        # n为表的最大数量，left_table和right_table都是长度为2n的one-hot编码
        feature_type = np.zeros(len(ALL_TYPES))
        feature_type[ALL_TYPES.index(plan["Node Type"])] = 1
        feature = np.concatenate((feature_type, self.to_feature_cost(plan)))
        feature = torch.tensor(feature,device = cfg.device,dtype = torch.float32).reshape(-1,cfg.input_size)
        return feature
    def to_feature_scan(self,plan):
        feature_type = np.zeros(len(ALL_TYPES))
        rel_name = plan["Relation Name"]
        feature_type[ALL_TYPES.index(plan["Node Type"])] = 1
        feature = np.concatenate((feature_type, self.to_feature_cost(plan)))
        feature = torch.tensor(feature,device = cfg.device,dtype = torch.float32).reshape(-1,cfg.input_size)
        return (feature, torch.tensor(self.to_table_id(plan), device = cfg.device,dtype = torch.long))
    def plan_to_feature_tree(self, plan):
        if "Plan" in plan:
            plan = plan["Plan"]
        children = plan["Plan"] if "Plan" in plan else (plan["Plans"] if "Plans" in plan else [])
        # len(children) == 1 表示node type为gather，是并行时产生的结果，没有实际意义
        if len(children) == 1:
            return self.plan_to_feature_tree(children[0])
        if plan["Node Type"] in JOIN_TYPES:
            left = self.plan_to_feature_tree(children[0])
            right = self.plan_to_feature_tree(children[1])
            pos= self.to_feature_join(plan)
            return (pos,left,right)
        if plan["Node Type"] in LEAF_TYPES:
            pos= self.to_feature_scan(plan)
            return pos


class ValueExtractor:
    def __init__(self,max_value = 20):
        self.max_value = max_value
    # def encode(self,v):
    #     return np.log(self.offset+v)/np.log(2)/self.max_value
    # def decode(self,v):
    #     # v=-(v*v<0)
    #     return np.exp(v*self.max_value*np.log(2))#-self.offset
    def encode(self,v):
        return torch.tensor([[int(np.log(2+v)/np.log(cfg.max_time_out)*200)/200.]], device = cfg.device, dtype = torch.float)
        return int(np.log(self.offset+v)/np.log(config.max_time_out)*200)/200.
    def decode(self,v):
        # v=-(v*v<0)
        # return np.exp(v/2*np.log(config.max_time_out))#-self.offset
        return np.exp(v*np.log(cfg.max_time_out))#-self.offset
    def cost_encode(self,v,min_cost,max_cost):
        return (v-min_cost)/(max_cost-min_cost)
    def cost_decode(self,v,min_cost,max_cost):
        return (max_cost-min_cost)*v+min_cost
    def latency_encode(self,v,min_latency,max_latency):
        return (v-min_latency)/(max_latency-min_latency)
    def latency_decode(self,v,min_latency,max_latency):
        return (max_latency-min_latency)*v+min_latency
    def rows_encode(self,v,min_cost,max_cost):
        return (v-min_cost)/(max_cost-min_cost)
    def rows_decode(self,v,min_cost,max_cost):
        return (max_cost-min_cost)*v+min_cost                                 

