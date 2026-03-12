import os
import logging
from datetime import datetime

class Logger:
    def __init__(self,
                 log_dir=None,
                 log_name=None,
                 ):

        self.log_dir = log_dir
        self.make_directory(self.log_dir)

        self.log_name = log_name
        self.log_filename = self.generate_log_filename(self.log_name)
        self.log_filepath = os.path.join(self.log_dir, self.log_filename)

        self.log = logging.getLogger("")
        self.log.setLevel(logging.INFO)

        file_handler = logging.FileHandler(self.log_filepath)
        console_handler = logging.StreamHandler()

        file_handler.setLevel(logging.INFO)
        console_handler.setLevel(logging.INFO)

        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)

        self.log.addHandler(file_handler)
        self.log.addHandler(console_handler)
        self.log.critical(f">>> Runtime Name: {self.log_name}")

    @staticmethod
    def make_directory(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def generate_log_filename(runtime_name):
        now = datetime.now()
        timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')
        return f"{timestamp}_{runtime_name}.log"

    def debug(self, message):
        self.log.debug(message)

    def info(self, message):
        self.log.info(message)

    def warning(self, message):
        self.log.warning(message)

    def error(self, message):
        self.log.error(message)

    def critical(self, message):
        self.log.critical(message)