"""
Test script to verify that all network architectures work correctly.
Tests each network (iTransformer, RNN, TCN, iSpikformer, SpikeRNN, SpikeTCN) 
with 1 epoch to ensure they run without errors.

Usage:
    # Run all tests
    python test/test_networks.py
    
    # Run only failed tests from previous run
    python test/test_networks.py --retry
    
    # Force run all tests (ignore cache)
    python test/test_networks.py --all
    
    # Clear test cache
    python test/test_networks.py --clear

    # Run only a specific network test
    python test/test_networks.py --only iSpikformer
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Define test configurations for all datasets
ALL_DATASETS = [
    ("electricity", "electricity"),
    ("etth1", "etth1"),
    ("etth2", "etth2"),
    ("metr-la", "metr-la"),
    ("solar", "solar"),
    ("weather", "weather"),
]

def get_test_configs(mode="simple"):
    configs = {}
    if mode == "simple":
        # Only etth1 for each model
        configs = {
            "iTransformer": "exp/forecast/ann/itransformer/itransformer_etth1.yml",
            "RNN": "exp/forecast/ann/rnn/rnn_etth1.yml",
            "TCN": "exp/forecast/ann/tcn/tcn_etth1.yml",
            "iSpikformer": "exp/forecast/snn/ispikformer/ispikformer_etth1.yml",
            "SpikeRNN": "exp/forecast/snn/spikernn/spikernn_etth1.yml",
            "SpikeTCN": "exp/forecast/snn/spiketcn/spiketcn_etth1.yml",
        }
    elif mode == "extended":
        # All 6 datasets for each model
        for model, prefix, yml_prefix in [
            ("iTransformer", "ann/itransformer/itransformer_", "itransformer"),
            ("RNN", "ann/rnn/rnn_", "rnn"),
            ("TCN", "ann/tcn/tcn_", "tcn"),
            ("iSpikformer", "snn/ispikformer/ispikformer_", "ispikformer"),
            ("SpikeRNN", "snn/spikernn/spikernn_", "spikernn"),
            ("SpikeTCN", "snn/spiketcn/spiketcn_", "spiketcn"),
        ]:
            for ds, dsfile in ALL_DATASETS:
                key = f"{model}_{ds}"
                configs[key] = f"exp/forecast/{prefix}{dsfile}.yml"
    return configs

# For each network, also test with clustering enabled if supported
CLUSTERING_SUPPORTED = {"SpikeRNN", "SpikeTCN", "iSpikformer"}

def get_test_variants(mode="simple"):
    configs = get_test_configs(mode)
    variants = {}
    for name, config in configs.items():
        base_model = name.split("_")[0] if "_" in name else name
        variants[name] = {"config": config, "use_cluster": False}
        if base_model in CLUSTERING_SUPPORTED:
            variants[name + "_cluster"] = {"config": config, "use_cluster": True}
    return variants


# Result file for test results
RESULT_FILE = Path(__file__).parent / ".test_result.json"


def load_cache():
    """Load previous test results from cache file."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(results):
    """Save test results to cache file."""
    cache_data = {
        "timestamp": datetime.now().isoformat(),
        "results": results
    }
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache_data, f, indent=2)


def clear_cache():
    """Clear the test cache file."""
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
        print(f"✅ Cache cleared: {CACHE_FILE}")
    else:
        print("ℹ️  No cache file found")



def run_network_test(network_name: str, config_path: str, use_cluster: bool = False, verbose: bool = False) -> bool:
    """
    Test a single network with 1 epoch.
    If use_cluster is True, override clustering-related config via CLI.
    """
    print(f"\n{'='*60}")
    print(f"Testing {network_name}...{' (with clustering)' if use_cluster else ''}")
    print(f"Config: {config_path}")
    print(f"{'='*60}")
    # No gpu_memory handling here; handled in tsforecast.py

    cse_dir = Path(__file__).parent.parent
    full_config_path = cse_dir / config_path

    if not full_config_path.exists():
        print(f"❌ Config file not found: {full_config_path}")
        return False

    original_dir = os.getcwd()
    os.chdir(cse_dir)

    import traceback
    try:
        cmd = [
            sys.executable,
            "-m",
            "SeqSNN.entry.tsforecast",
            str(config_path),
            "--runner.max_epoches", "1",
            "--runner.early_stop", "1",
            "--runtime.output_dir", f"./output/test_{network_name.lower()}"
        ]
        if use_cluster:
            cmd += [
                "--network.use_cluster", "True",
                "--network.n_cluster", "3",
                "--network.d_model", "64",
                "--network.cluster_method", "attention"
            ]
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3000
        )
        if result.returncode == 0:
            print(f"✅ {network_name} test PASSED")
            if verbose:
                print(f"--- STDOUT ---\n{result.stdout}")
                print(f"--- STDERR ---\n{result.stderr}")
            return True
        else:
            print(f"❌ {network_name} test FAILED")
            if verbose:
                print(f"--- STDERR ---\n{result.stderr}")
                print(f"--- STDOUT ---\n{result.stdout}")
            return False
    except subprocess.TimeoutExpired:
        print(f"❌ {network_name} test TIMEOUT (exceeded 5 minutes)")
        if verbose:
            traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ {network_name} test FAILED with exception: {e}")
        if verbose:
            traceback.print_exc()
        return False
    finally:
        os.chdir(original_dir)



def main():
    """Run all network tests, including clustering variants. Use --mode simple|extended."""
    parser = argparse.ArgumentParser(description="Test network architectures")
    parser.add_argument("--retry", action="store_true", 
                       help="Run only failed tests from previous run")
    parser.add_argument("--all", action="store_true",
                       help="Force run all tests (ignore cache)")
    parser.add_argument("--clear", action="store_true",
                       help="Clear test cache and exit")
    parser.add_argument("--only", type=str, metavar="NETWORK", help="Test only the specified network (by name, add _cluster for clustering variant)")
    parser.add_argument("--mode", type=str, choices=["simple", "extended"], default="simple", help="Test mode: simple (etth1 only) or extended (all datasets)")
    parser.add_argument("--verbose", default=False, action="store_true", help="Show traceback and subprocess output for all tests")
    args = parser.parse_args()

    if args.clear:
        clear_cache()
        return 0

    import json
    results = {}
    if RESULT_FILE.exists():
        try:
            with open(RESULT_FILE, 'r') as f:
                results = json.load(f)
        except Exception:
            results = {}
    previous_results = {k: v.get('result', False) for k, v in results.items()}

    test_variants = get_test_variants(args.mode)

    if args.only:
        if args.only not in test_variants:
            print(f"\n❌ Unknown network: {args.only}")
            print(f"Available: {', '.join(test_variants.keys())}")
            return 1
        tests_to_run = {args.only: test_variants[args.only]}
        print(f"\nRunning only: {args.only}")
    elif args.retry and previous_results:
        tests_to_run = {
            name: variant for name, variant in test_variants.items()
            if not previous_results.get(name, False)
        }
        if not tests_to_run:
            print("\n✅ All tests passed in previous run. Nothing to retry.")
            print("Use --all to run all tests again.")
            return 0
        print("\n" + "="*60)
        print("Retrying Failed Tests")
        print("="*60)
    elif args.all or not previous_results:
        tests_to_run = test_variants
        print("\n" + "="*60)
        print(f"Starting Network Architecture Tests (mode: {args.mode}, with clustering variants)")
        print("="*60)
    else:
        tests_to_run = {
            name: variant for name, variant in test_variants.items()
            if not previous_results.get(name, False)
        }
        skipped = len(test_variants) - len(tests_to_run)
        print("\n" + "="*60)
        print(f"Starting Network Architecture Tests (mode: {args.mode}, with clustering variants)")
        print(f"Skipping {skipped} previously passed test(s)")
        print("Use --all to force run all tests")
        print("="*60)
        if not tests_to_run:
            print("\n✅ All tests passed in previous run.")
            print("Use --all to run all tests again.")
            return 0

    from datetime import datetime
    for network_name, variant in tests_to_run.items():
        success = run_network_test(network_name, variant["config"], use_cluster=variant["use_cluster"], verbose=args.verbose)
        prev = results.get(network_name, {})
        entry = dict(prev)
        entry["timestamp"] = datetime.now().isoformat()
        entry["result"] = success
        # Do not touch gpu_memory or any other field
        results[network_name] = entry

    # Save new structure
    with open(RESULT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    passed = sum(1 for v in results.values() if isinstance(v, dict) and v.get('result', False))
    total = len(results)

    for network_name in test_variants.keys():
        success = results.get(network_name, False)
        ran_now = network_name in tests_to_run
        status = "✅ PASSED" if success else "❌ FAILED"
        marker = " (ran now)" if ran_now else " (cached)"
        print(f"{network_name:25s}: {status}{marker if not args.all else ''}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        failed_tests = [name for name, success in results.items() if not success]
        print(f"\n⚠️  {total - passed} test(s) failed: {', '.join(failed_tests)}")
        print("\nRun with --retry to test only failed networks")
        return 1


if __name__ == "__main__":
    sys.exit(main())
