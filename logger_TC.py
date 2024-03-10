import logging
import time


class Logger:
    def __init__(self, log_file):
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s',
                            datefmt='%Y-%m-%dT%H:%M:%S%z', encoding='utf8')
        self.start_time = time.time()

    @staticmethod
    def log_start():
        logging.info("Программа запущена")

    def log_end(self):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        logging.info(f"Программа завершена. Время выполнения: {elapsed_time:.2f} секунд.")

    @staticmethod
    def log_settings(ticker, date, balance, interval, days):
        logging.info(f"Тикер: {ticker}")
        logging.info(f"Дата: {date}")
        logging.info(f"Баланс: {balance}")
        logging.info(f"Интервал: {interval}")
        logging.info(f"Дни: {days}")

    @staticmethod
    def log_results(results):
        logging.info(f"Результаты работы программы: {results}")

    @staticmethod
    def log_error(error_message):
        logging.error(f"Ошибка: {error_message}")