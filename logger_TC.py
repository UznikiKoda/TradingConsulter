import logging
import time
import psycopg2


class Logger:
    def __init__(self, host, db, user, password, port):
        self.start_time = time.time()
        self.conn = psycopg2.connect(host=host, dbname=db, user=user, password=password, port=port)
        self.cursor = self.conn.cursor()

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
        #222222
    def log_postgres(self, ticker, date_time, balance, interval, days):
        self.cursor.execute("""
                   INSERT INTO public.logs
                   (ticker, date_time, balance , interval, days)
                   VALUES (%s, %s, %s, %s, %s)
           """,  (ticker, date_time, balance, interval, days))
        self.conn.commit()