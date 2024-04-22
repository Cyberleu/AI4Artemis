import socket
import psycopg2
class ArtemisConnector:
    def __init__(self, HOST,PORT):
        self.host = HOST
        self.port = PORT
        self.sql = ""
        self.joinOrder = []
        self.predicates = []
    def recieve(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen()
            conn, addr = s.accept()
            next_sql = 0
            next_joinorder = 0
            next_predicates = 0
            with conn:
                print('Connected by', addr)
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    if(data.decode()[0] == "0"):
                        next_sql = 1
                    elif(data.decode()[0] == "1"):
                        next_joinorder = 1
                        next_sql = 0
                    elif(data.decode()[0] == "2"):
                        next_predicates = 1
                        next_joinorder = 0
                    elif(next_sql == 1):
                        self.sql = data.decode()
                    elif(next_joinorder == 1):
                        self.joinOrder.append(data.decode())
                    elif(next_predicates == 1):
                        self.predicates.append(data.decode())
    def print(self):
        print("sql" + self.sql)
class PGConnector:
    def __init__(self, dbname='', user='', password='', host='', port=''):

        self.con = psycopg2.connect(database=dbname, user=user,
                                    password=password, host=host, port=port)
        self.cur = self.con.cursor()
    def getPGPlan(self,sql):
        self.cur.execute("explain (COSTS, FORMAT JSON, ANALYSE) "+sql)
        rows = self.cur.fetchall()
        PGPlan = rows[0][0][0]
        return PGPlan
    def getPGLatency(self,sql):
        return self.getPGPlan(sql)['Execution Time']
    def getPGSelectivity(self,table,predicates):
        totalQuery = "select * from "+table+";"
        self.cur.execute("EXPLAIN "+totalQuery)
        rows = self.cur.fetchall()[0][0]
        total_rows = int(rows.split("rows=")[-1].split(" ")[0])
        resQuery = "select * from "+table+" Where "+predicates+";"
        self.cur.execute("EXPLAIN  "+resQuery)
        rows = self.cur.fetchall()[0][0]
        select_rows = int(rows.split("rows=")[-1].split(" ")[0])
        return select_rows/total_rows
    def getAllTables(self):
        allTables = []
        table2index = {}
        self.cur.execute("select * from pg_tables;")
        rows = self.cur.fetchall()
        for row in rows:
            if row[0] == "public":
                allTables.append(row[1])
            else :
                break
        count = 0
        for table in allTables:
            table2index[table] = count
            count = count+1
        return table2index, allTables