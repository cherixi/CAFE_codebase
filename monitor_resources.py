#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每 1 秒记录一次系统内存、/dev/shm 使用情况和各个 GPU 显存占用，输出到 txt 文件。
依赖：
    pip install psutil
环境：
    Linux + NVIDIA GPU（需要 nvidia-smi 命令可用）
"""

import time
import datetime
import subprocess
import psutil
import os

LOG_FILE = "resource_usage_log.txt"  # 日志文件名
INTERVAL = 1.0                       # 采样间隔（秒）


def get_gpu_memory():
    """
    使用 nvidia-smi 获取各 GPU 显存 (MB) 使用情况。
    返回: (gpu_list, error)
        gpu_list: [(used_mb, total_mb), ...]
        error: 如果出错则为字符串，否则为 None
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
    except Exception as e:
        return [], str(e)

    gpus = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        used_str, total_str = [x.strip() for x in line.split(",")]
        try:
            used = int(used_str)
            total = int(total_str)
        except ValueError:
            # 避免解析异常
            continue
        gpus.append((used, total))
    return gpus, None


def get_shm_usage():
    """
    获取 /dev/shm 使用情况（一般是共享内存所在的 tmpfs）。
    返回: (used_mb, total_mb)
    """
    try:
        du = psutil.disk_usage("/dev/shm")
        used_mb = du.used // (1024 * 1024)
        total_mb = du.total // (1024 * 1024)
        return used_mb, total_mb
    except FileNotFoundError:
        # 某些系统可能没有 /dev/shm
        return None, None


def init_log():
    """如果日志文件不存在，写入表头。"""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            # 先初始化一次 GPU 列数，方便写 header
            gpus, err = get_gpu_memory()
            gpu_cols = []
            for i in range(len(gpus)):
                gpu_cols.append(f"gpu{i}_mem_used_MB")
                gpu_cols.append(f"gpu{i}_mem_total_MB")

            header = [
                "timestamp",
                "ram_used_MB",
                "ram_total_MB",
                "ram_percent",
                "shm_used_MB",
                "shm_total_MB",
            ] + gpu_cols

            f.write("\t".join(header) + "\n")


def main():
    print(f"开始记录资源使用情况，每 {INTERVAL} 秒一次，输出到 {LOG_FILE}")
    init_log()

    try:
        while True:
            # 时间戳
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 物理内存
            vm = psutil.virtual_memory()
            ram_used_mb = vm.used // (1024 * 1024)
            ram_total_mb = vm.total // (1024 * 1024)
            ram_percent = vm.percent

            # shm (/dev/shm)
            shm_used_mb, shm_total_mb = get_shm_usage()

            # GPU 显存
            gpus, gpu_err = get_gpu_memory()

            row = [
                ts,
                str(ram_used_mb),
                str(ram_total_mb),
                f"{ram_percent:.1f}",
                str(shm_used_mb) if shm_used_mb is not None else "NA",
                str(shm_total_mb) if shm_total_mb is not None else "NA",
            ]

            for used, total in gpus:
                row.append(str(used))
                row.append(str(total))

            # 如果 nvidia-smi 有问题，也可以在这里记录一下错误信息（可选）
            # 比如扩展一列 gpu_error，不过这里先简单忽略

            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write("\t".join(row) + "\n")

            time.sleep(INTERVAL)

    except KeyboardInterrupt:
        print("\n收到中断信号，停止记录。")


if __name__ == "__main__":
    main()
