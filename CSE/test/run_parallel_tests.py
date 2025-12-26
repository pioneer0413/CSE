import os
import subprocess
import sys
import time
from pathlib import Path

def get_least_used_gpu():
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=memory.used',
                '--format=csv,nounits,noheader'
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        mem_used = [int(x) for x in result.stdout.strip().split('\n')]
        return int(min(range(len(mem_used)), key=lambda i: mem_used[i]))
    except Exception as e:
        print(f"[Warning] Could not determine least used GPU, using device 0. Reason: {e}")
        return 0

# 모델, 데이터셋, seed 목록 정의
MODELS = [
    ("iTransformer", "exp/forecast/ann/itransformer/itransformer_{}.yml", []),
    ("RNN", "exp/forecast/ann/rnn/rnn_{}.yml", []),
    ("TCN", "exp/forecast/ann/tcn/tcn_{}.yml", []),
    ("iSpikformer", "exp/forecast/snn/ispikformer/ispikformer_{}.yml", []),
    ("SpikeRNN", "exp/forecast/snn/spikernn/spikernn_{}.yml", []),
    ("SpikeTCN", "exp/forecast/snn/spiketcn/spiketcn_{}.yml", []),
    ("iSpikformer", "exp/forecast/snn/ispikformer/ispikformer_{}.yml", ["--network.use_cluster", "True", "--network.n_cluster", "3"]),
    ("SpikeRNN", "exp/forecast/snn/spikernn/spikernn_{}.yml", ["--network.use_cluster", "True", "--network.n_cluster", "3"]),
    ("SpikeTCN", "exp/forecast/snn/spiketcn/spiketcn_{}.yml", ["--network.use_cluster", "True", "--network.n_cluster", "3"]),
]
DATASETS = ["electricity", "etth1", "etth2", "weather"] # "solar", "metr-la"
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

# 실험 조합 생성
EXPERIMENTS = []
for model, config_tpl, extra_args in MODELS:
    for dataset in DATASETS:
        for seed in SEEDS:
            config_path = config_tpl.format(dataset)
            out_model = model.lower()
            use_cluster = any(
                (arg == "--network.use_cluster" and val.lower() == "true")
                for arg, val in zip(extra_args[::2], extra_args[1::2])
            )
            cluster_dir = "cluster" if use_cluster else "noncluster"
            out_dataset = dataset.lower()
            output_dir = f"./output/{out_model}/{cluster_dir}/{out_dataset}/seed{seed}"
            args = extra_args + ["--runtime.seed", str(seed)]
            EXPERIMENTS.append((model, config_path, args, output_dir))

MAX_PARALLEL = 4  # 동시에 실행할 최대 프로세스 수
processes = []
results = {}

for exp in EXPERIMENTS:
    net, config_path, extra_args, output_dir = exp
    gpu_id = get_least_used_gpu()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Ensure output_dir's parent exists
    out_path = Path(output_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m", "SeqSNN.entry.tsforecast",
        config_path,
        "--runner.max_epoches", "300",
        "--runner.early_stop", "30",
        "--runtime.output_dir", output_dir
    ] + extra_args
    print(f"Launching {net} on GPU {gpu_id}: {' '.join(cmd)}")
    p = subprocess.Popen(cmd, env=env)
    processes.append((net, p, gpu_id, cmd, output_dir))
    time.sleep(10)  # 10초 대기: GPU context 초기화 및 할당 대기
    if len(processes) >= MAX_PARALLEL:
        for net_name, proc, gpu_used, cmdline, outdir in processes:
            proc.wait()
            results[(net_name, outdir)] = proc.returncode
            print(f"{net_name} (GPU {gpu_used}) finished with code {proc.returncode} (output: {outdir})")
        processes = []

for net_name, proc, gpu_used, cmdline, outdir in processes:
    proc.wait()
    results[(net_name, outdir)] = proc.returncode
    print(f"{net_name} (GPU {gpu_used}) finished with code {proc.returncode} (output: {outdir})")

print("\nSummary:")
for (net, outdir), code in results.items():
    print(f"{net} [{outdir}]: {'✅ PASS' if code == 0 else '❌ FAIL'}")
