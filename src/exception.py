# exception.py
import sys
from logger import logging  # Correct import for a module in the same directory

def error_message_detail(error, error_detail):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in Python script '{file_name}', line {line_number}: {error}"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message

try:
    a = 10 / 0
except Exception as e:
    logging.info("devide zero")
    raise CustomException(str(e), sys)
