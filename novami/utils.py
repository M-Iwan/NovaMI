import datetime

def log(message: str):
    log_time = str(datetime.datetime.now()).split('.')[0]
    print(f'{log_time}: {message}')