class Config:
    def __init__(self):
        self.pghost = "127.0.0.1"
        self.pgport = "5432"
        self.server = "9999"
        self.username = "cyberleu"
        self.password = ""
        self.database = "artemis"
        self.hidden_size = 64
        self.device = "cpu"
        self.max_time_out = 120*1000
        self.query_path = "query.txt"
        self.max_table_num = 6
        self.max_column = 200
        self.split_rate = 0.2
        self.input_size = 7+2
        