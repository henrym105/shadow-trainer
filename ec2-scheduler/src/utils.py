def get_current_time_in_timezone(timezone: str) -> str:
    from datetime import datetime
    import pytz

    tz = pytz.timezone(timezone)
    return datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')

def is_time_to_start(current_time: str) -> bool:
    return current_time.endswith('09:00:00')

def is_time_to_stop(current_time: str) -> bool:
    return current_time.endswith('22:00:00')

def log_event(message: str) -> None:
    from datetime import datetime
    with open('scheduler.log', 'a') as log_file:
        log_file.write(f"{datetime.now()}: {message}\n")