import logging

class Logger:
    def __init__(self, event_name, log_file='../output/historical_logs.log', log_level=logging.DEBUG):
        self.logger = logging.getLogger()
        self.logger.setLevel(log_level)
        self.event_name = event_name

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

    def debug(self, message):
        self.logger.debug(f"{message} for {self.event_name}")

    def info(self, message):
        self.logger.info(f"{message} for {self.event_name}")

    def warning(self, message):
        self.logger.warning(f"{message} for {self.event_name}")

    def error(self, message):
        self.logger.error(f"{message} for {self.event_name}")

    def critical(self, message):
        self.logger.critical(f"{message} for {self.event_name}")
