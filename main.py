from indicators.indicator import Indicator
from logger_TC import Logger
import os
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    log = Logger(
        host=os.getenv('POSTGRES_HOST'),
        db=os.getenv('POSTGRES_DB'),
        user=os.getenv('POSTGRES_USER'),
        password=os.getenv('POSTGRES_PASSWORD'),
        port=os.getenv('POSTGRES_PORT')
    )
    
    res = Indicator(log,'GAZP', '06.08.2020', balance=100000, interval='day', days=365)
    res.best_solution()
    log.log_end()

if __name__ == '__main__':
    main()
