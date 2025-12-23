import os
import subprocess
import sys


def main():
    # 通过 torchrun 启动 main_dp.py 跑一个短程，检查退出码
    nproc = 2
    cmd = [
        sys.executable,
        "-m",
        "torchrun",
        f"--nproc_per_node={nproc}",
        "--master_addr=127.0.0.1",
        "--master_port=29500",
        "main_dp.py",
    ]
    env = os.environ.copy()
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        raise SystemExit(f"DDP run failed with code {res.returncode}")


if __name__ == "__main__":
    main()
