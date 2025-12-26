import warnings

from utilsd import get_output_dir, get_checkpoint_dir, setup_experiment
from utilsd.experiment import print_config
from utilsd.config import PythonConfig, RegistryConfig, RuntimeConfig, configclass

from SeqSNN.dataset import DATASETS
from SeqSNN.runner import RUNNERS
from SeqSNN.network import NETWORKS

import time

warnings.filterwarnings("ignore")


@configclass
class SeqSNNConfig(PythonConfig):
    data: RegistryConfig[DATASETS]
    network: RegistryConfig[NETWORKS]
    runner: RegistryConfig[RUNNERS]
    runtime: RuntimeConfig = RuntimeConfig()


def run_train(config):
    setup_experiment(config.runtime)
    print_config(config)

    '''
    메모리 할당 공간이 가장 많은 GPU id 식별
    '''
    min_gpu_id = config.network.device if hasattr(config.network, 'device') else 0

    trainset = config.data.build(dataset_name="train")
    validset = config.data.build(dataset_name="valid")
    testset = config.data.build(dataset_name="test")
    network = config.network.build(
        input_size=trainset.num_variables, max_length=trainset.max_seq_len, 
        device=min_gpu_id
    )
    runner = config.runner.build(
        network=network,
        output_dir=get_output_dir(),
        checkpoint_dir=get_checkpoint_dir(),
        out_size= trainset.horizon if config.data.is_vsts else trainset.num_classes, 
        device=min_gpu_id
    )
    runner.fit(trainset, validset, testset)
    runner.predict(trainset, "train")
    runner.predict(validset, "valid")
    runner.predict(testset, "test")

    # If max_epoches==1, save GPU max memory usage to test/.test_gpu.json
    try:
        max_epochs = getattr(config.runner, 'max_epoches', None)
        if max_epochs == 1 and hasattr(config, 'runtime'):
            import torch
            import json
            from pathlib import Path
            gpu_path = Path(__file__).parent.parent.parent / 'test' / '.test_gpu.json'
            gpu_mem = 0
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.max_memory_allocated() // (1024 * 1024)
                print('[Info] GPU max memory allocated: {} MB'.format(gpu_mem))
            # Robust net_name extraction from output_dir
            output_dir = str(getattr(config.runtime, 'output_dir', ''))
            net_name = 'unknown'
            if output_dir:
                import os
                import re
                base = os.path.basename(output_dir.rstrip('/'))
                # Match test_xxx(_cluster)? or output/xxx(_cluster)?
                m = re.match(r'(?:test_|output/)?([a-zA-Z0-9]+(?:_cluster)?)', base)
                if m:
                    net_name = m.group(1)
                else:
                    net_name = base
            from datetime import datetime
            # Save as {net_name: {timestamp, gpu_memory}}
            gpu_data = {}
            if gpu_path.exists():
                try:
                    with open(gpu_path, 'r') as f:
                        gpu_data = json.load(f)
                except Exception:
                    gpu_data = {}
            gpu_entry = {
                "timestamp": datetime.now().isoformat(),
                "gpu_memory": gpu_mem
            }
            gpu_data[net_name] = gpu_entry
            with open(gpu_path, 'w') as f:
                json.dump(gpu_data, f, indent=2)
                f.flush()
    except Exception as e:
        print(f"[Warning] Failed to save GPU memory usage: {e}")

if __name__ == "__main__":
    _config = SeqSNNConfig.fromcli()
    run_train(_config)
