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

# Define test configurations
TEST_CONFIGS = {
    "iTransformer": "exp/forecast/ann/itransformer/itransformer_etth1.yml",
    "RNN": "exp/forecast/ann/rnn/rnn_etth1.yml",
    "TCN": "exp/forecast/ann/tcn/tcn_etth1.yml",
    "iSpikformer": "exp/forecast/snn/ispikformer/ispikformer_etth1.yml",
    "SpikeRNN": "exp/forecast/snn/spikernn/spikernn_etth1.yml",
    "SpikeTCN": "exp/forecast/snn/spiketcn/spiketcn_etth1.yml",
}

# Cache file for test results
CACHE_FILE = Path(__file__).parent / ".test_cache.json"


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


def run_network_test(network_name: str, config_path: str) -> bool:
    """
    Test a single network with 1 epoch.
    
    Args:
        network_name: Name of the network being tested
        config_path: Path to the configuration file
        
    Returns:
        True if test passed, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Testing {network_name}...")
    print(f"Config: {config_path}")
    print(f"{'='*60}")
    
    # Get the CSE directory (parent of test directory)
    cse_dir = Path(__file__).parent.parent
    full_config_path = cse_dir / config_path
    
    if not full_config_path.exists():
        print(f"❌ Config file not found: {full_config_path}")
        return False
    
    # Change to CSE directory
    original_dir = os.getcwd()
    os.chdir(cse_dir)
    
    try:
        # Run the network with modified settings for testing
        cmd = [
            sys.executable,
            "-m",
            "SeqSNN.entry.tsforecast",
            str(config_path),
            "--runner.max_epoches", "1",
            "--runner.early_stop", "1",
            "--runtime.output_dir", f"./output/test_{network_name.lower()}",
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"✅ {network_name} test PASSED")
            return True
        else:
            print(f"❌ {network_name} test FAILED")
            print(f"STDERR:\n{result.stderr}")
            print(f"STDOUT:\n{result.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ {network_name} test TIMEOUT (exceeded 5 minutes)")
        return False
    except Exception as e:
        print(f"❌ {network_name} test FAILED with exception: {e}")
        return False
    finally:
        os.chdir(original_dir)


def main():
    """Run all network tests."""
    parser = argparse.ArgumentParser(description="Test network architectures")
    parser.add_argument("--retry", action="store_true", 
                       help="Run only failed tests from previous run")
    parser.add_argument("--all", action="store_true",
                       help="Force run all tests (ignore cache)")
    parser.add_argument("--clear", action="store_true",
                       help="Clear test cache and exit")
    parser.add_argument("--only", type=str, metavar="NETWORK", help="Test only the specified network (by name)")
    args = parser.parse_args()
    
    # Handle cache clearing
    if args.clear:
        clear_cache()
        return 0
    
    # Load previous results
    cache = load_cache()
    previous_results = cache.get("results", {})
    
    # Determine which tests to run
    if args.only:
        if args.only not in TEST_CONFIGS:
            print(f"\n❌ Unknown network: {args.only}")
            print(f"Available: {', '.join(TEST_CONFIGS.keys())}")
            return 1
        tests_to_run = {args.only: TEST_CONFIGS[args.only]}
        print(f"\nRunning only: {args.only}")
    elif args.retry and previous_results:
        # Run only failed tests
        tests_to_run = {
            name: config for name, config in TEST_CONFIGS.items()
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
        # Run all tests
        tests_to_run = TEST_CONFIGS
        print("\n" + "="*60)
        print("Starting Network Architecture Tests")
        print("="*60)
    else:
        # Default: skip passed tests, run failed/new tests
        tests_to_run = {
            name: config for name, config in TEST_CONFIGS.items()
            if not previous_results.get(name, False)
        }
        skipped = len(TEST_CONFIGS) - len(tests_to_run)
        print("\n" + "="*60)
        print("Starting Network Architecture Tests")
        print(f"Skipping {skipped} previously passed test(s)")
        print("Use --all to force run all tests")
        print("="*60)
        
        if not tests_to_run:
            print("\n✅ All tests passed in previous run.")
            print("Use --all to run all tests again.")
            return 0
    
    # Run tests
    results = dict(previous_results)  # Start with previous results
    
    for network_name, config_path in tests_to_run.items():
        success = run_network_test(network_name, config_path)
        results[network_name] = success
    
    # Save results to cache
    save_cache(results)
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for network_name in TEST_CONFIGS.keys():
        success = results.get(network_name, False)
        ran_now = network_name in tests_to_run
        status = "✅ PASSED" if success else "❌ FAILED"
        marker = " (ran now)" if ran_now else " (cached)"
        print(f"{network_name:20s}: {status}{marker if not args.all else ''}")
    
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
