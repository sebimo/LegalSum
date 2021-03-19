import sqlite3
from enum import Enum
from datetime import datetime
from pathlib import Path
import numpy

from torch.utils.tensorboard import SummaryWriter

class Type(Enum):
    TEXT = 1
    INT = 2
    FLOAT = 3

class Logger:

    def __init__(self, database: Path):
        print("Starting experiment database:", database)
        self.conn = sqlite3.connect(database)
        self.c = self.conn.cursor()
        self.__setup_experiment_db__()
        self.__setup_epoch_db__()
        self.experiment = None
        self.exp_num = None
        self.epoch = 0
        self.on = True

    def add_parameter(self, name: str, type: Type):
        type_str = "TEXT" if type == Type.TEXT else ("INTEGER" if type == Type.INT else "REAL")
        request = "ALTER TABLE experiment ADD COLUMN "+name+" "+type_str+" DEFAULT NULL"
        try:
            self.c.execute(request)
            print("Add parameter: "+name)
        except sqlite3.OperationalError:
            pass

    def add_perf_metric(self, name: str, type: Type):
        type_str = "TEXT" if type == Type.TEXT else ("INTEGER" if type == Type.INT else "REAL")
        request = "ALTER TABLE epoch ADD COLUMN "+name+" "+type_str+" DEFAULT NULL"
        try:
            self.c.execute(request)
            print("Add performance metric: "+name)
        except sqlite3.OperationalError:
            pass

    def start_experiment(self, parameters: dict={}):
        time = datetime.now()
        self.experiment = time.strftime("%d_%m_%Y__%H%M%S")
        if self.on:
            columns = [i[1] for i in self.c.execute("PRAGMA table_info(experiment)")]
            request = "INSERT INTO experiment ("
            for key in parameters:
                if key not in columns:
                    raise sqlite3.OperationalError("Unknown column: "+key)
                request += str(key)+","
            request += "date) VALUES ("
            for key in parameters:
                if isinstance(parameters[key], str):
                    request += "\""+str(parameters[key])+"\","
                else:
                    request += str(parameters[key])+","
            request += "\""+str(self.experiment)+"\");"
            print(request)
            self.c.execute(request)
            self.conn.commit()

            comment = []
            for key in parameters:
                comment.append(str(key))
                comment.append(str(parameters[key]))
            comment = "_".join(comment)
            self.writer = SummaryWriter(log_dir=Path("logging")/"runs"/(self.experiment+"_"+comment))

            self.c.execute("SELECT rowid FROM experiment WHERE date=\""+self.experiment+"\";")
            self.exp_num = self.c.fetchone()[0]
            self.epoch = 0

    def log_epoch(self, metrics: dict):
        if self.on:
            if self.exp_num is None or self.experiment is None:
                raise sqlite3.OperationalError("Experiment not started.")
            self.epoch += 1
            columns = [i[1] for i in self.c.execute("PRAGMA table_info(epoch)")]
            request = "INSERT INTO epoch ("
            for key in metrics:
                if key not in columns:
                    raise sqlite3.OperationalError("Unknown column: "+key)
                request += str(key)+","
            request += "experiment,epoch) VALUES ("
            for key in metrics:
                request += str(metrics[key])+","
            request += str(self.exp_num)+","+str(self.epoch)+");"
            for key in metrics:
                self.writer.add_scalar(key, metrics[key], self.epoch)
            self.c.execute(request)
            self.conn.commit()

    def exp_info(self):
        if self.exp_num is None or self.experiment is None:
            raise KeyError("No experiment selected.")
        self.c.execute("SELECT * FROM experiment WHERE date=\""+str(self.experiment)+"\";")
        params = self.c.fetchone()
        self.c.execute("SELECT * FROM epoch WHERE experiment="+str(self.exp_num)+";")
        epochs = self.c.fetchall()
        return {"parameter": params, "epochs": epochs}

    def log_model(self, model):
        if self.on:
            self.writer.add_graph(model)

    def set_status(self, on: bool):
        self.on = on

    def __setup_experiment_db__(self):
        self.c.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='experiment'")
        if not self.c.fetchone()[0] == 1:
            print("Creating table: experiment")
            self.c.execute("CREATE TABLE experiment (date TEXT NOT NULL PRIMARY KEY)")
            self.conn.commit()

    def __setup_epoch_db__(self):
        self.c.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='epoch'")
        if not self.c.fetchone()[0] == 1:
            print("Creating table: epoch")
            self.c.execute("CREATE TABLE epoch (experiment INTEGER NOT NULL, epoch INTEGER NOT NULL)")
            self.conn.commit()

    def __del__(self):
        print("Closing connection to experiment database.")
        self.conn.commit()
        self.conn.close()

    def setup_extractive(self):
        print("Loggable parameters: model, lr, abstractive")
        self.add_parameter("model", Type.TEXT)
        self.add_parameter("lr", Type.TEXT)
        self.add_parameter("abstractive", Type.INT)
        self.add_parameter("embedding", Type.TEXT)
        self.add_parameter("attention", Type.TEXT)
        self.add_parameter("loss_type", Type.TEXT)
        self.add_parameter("target", Type.TEXT)
        self.add_perf_metric("train_loss", Type.FLOAT)
        self.add_perf_metric("val_loss", Type.FLOAT)
        self.add_perf_metric("train_F1", Type.FLOAT)
        self.add_perf_metric("train_Precision", Type.FLOAT)
        self.add_perf_metric("train_Recall", Type.FLOAT)
        self.add_perf_metric("val_F1", Type.FLOAT)
        self.add_perf_metric("val_Precision", Type.FLOAT)
        self.add_perf_metric("val_Recall", Type.FLOAT)

    def setup_abstractive(self):
        print("Loggable parameters: model, lr, abstractive")
        self.add_parameter("model", Type.TEXT)
        self.add_parameter("lr", Type.TEXT)
        self.add_parameter("abstractive", Type.INT)
        self.add_parameter("embedding", Type.TEXT)
        self.add_parameter("attention", Type.TEXT)
        self.add_parameter("loss_type", Type.TEXT)
        self.add_parameter("target", Type.TEXT)
        self.add_perf_metric("train_loss", Type.FLOAT)
        self.add_perf_metric("val_loss", Type.FLOAT)
        self.add_perf_metric("train_rouge-1", Type.FLOAT)
        self.add_perf_metric("train_rouge-2", Type.FLOAT)
        self.add_perf_metric("train_rouge-l", Type.FLOAT)
        self.add_perf_metric("val_rouge-1", Type.FLOAT)
        self.add_perf_metric("val_rouge-2", Type.FLOAT)
        self.add_perf_metric("val_rouge-l", Type.FLOAT)

if __name__ == "__main__":
    logger = Logger(database="test.db")
    logger.add_parameter("test_int", Type.INT)
    logger.add_perf_metric("acc", Type.FLOAT)
    try:
        logger.start_experiment(parameters={"test_int": 1, "hello": "world"})
    except sqlite3.OperationalError:
        print("Handled unknown column")
    try:
        logger.log_epoch(metrics={})
    except sqlite3.OperationalError:
        print("Unstarted experiment.")
    logger.start_experiment(parameters={"test_int": 1})
    for i in range(50):
        logger.log_epoch({"acc": 50+i*0.5 + (numpy.random.random() * 10)})
    print(logger.exp_info())