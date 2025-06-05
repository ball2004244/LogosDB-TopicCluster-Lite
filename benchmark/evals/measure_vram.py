from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetCount
import time


def get_vram_usage(gpu_id):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(gpu_id)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used, info.total


def log_vram_usage(log_file):
    nvmlInit()
    gpu_count = nvmlDeviceGetCount()
    while True:
        for gpu_id in range(gpu_count):
            used, total = get_vram_usage(gpu_id)
            used_gb = used / (1024 ** 3)
            total_gb = total / (1024 ** 3)
            with open(log_file, 'a') as f:
                log_entry = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - GPU {gpu_id} VRAM Usage: {used} bytes ({used_gb:.2f} GB / {total_gb:.2f} GB)\n"
                f.write(log_entry)
        time.sleep(5)  # every 5s


if __name__ == "__main__":
    log_file = "debug/overall_vram_usage.log"
    print('Start logging VRAM usage...')
    log_vram_usage(log_file)  # Measure overall VRAM
