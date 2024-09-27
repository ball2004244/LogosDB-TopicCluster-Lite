from datetime import datetime
from enum import Enum
from rich.console import Console
import pytz


class LogType(Enum):
    INFO = 'INFO'
    ERROR = 'ERROR'
    WARNING = 'WARNING'
    SUCCESS = 'SUCCESS'


console = Console()


def log(text: str, log_type: LogType, log_file: str = 'debug.log') -> None:
    # Get the current time in UTC
    utc_now = datetime.now(pytz.utc)
    # Convert the current time to EST
    est_now = utc_now.astimezone(pytz.timezone('US/Eastern'))

    formatted_time = est_now.strftime('%Y-%m-%d %H:%M:%S %Z')

    log_text = f'[{formatted_time}] {log_type.name}: {text}'

    if log_type == LogType.SUCCESS:
        console.print(f'[bold green]{log_text}[/bold green]')
    elif log_type == LogType.ERROR:
        console.print(f'[bold red]{log_text}[/bold red]')
    elif log_type == LogType.WARNING:
        console.print(f'[bold yellow]{log_text}[/bold yellow]')
    else:
        console.print(f'[bold]{log_text}[/bold]')

    with open(log_file, 'a') as f:
        f.write(f'{log_text}\n')


if __name__ == '__main__':
    log('Hello', log_type=LogType.INFO)
