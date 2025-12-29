import os
import json
import csv

def aggregate_results():
    output_dir = '/home/hwkang/CSE/CSE/output'
    results = []
    config_keys = None

    for root, dirs, files in os.walk(output_dir):
        if 'res.json' in files and root.endswith('checkpoints'):
            # Parse path to extract model, cluster, dataset, seed
            parts = root.split(os.sep)
            idx = parts.index('output')
            model = parts[idx + 1]
            cluster = parts[idx + 2]
            dataset = parts[idx + 3]
            seed = parts[idx + 4]
            
            # Read config.json
            config_path = os.path.join(os.path.dirname(root), 'config.json')
            config_data = {}
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    if config_keys is None:
                        config_keys = list(config_data.keys())
                except json.JSONDecodeError:
                    print(f"Error reading config {config_path}")
            
            # Read res.json
            res_path = os.path.join(root, 'res.json')
            try:
                with open(res_path, 'r') as f:
                    data = json.load(f)
                    test_data = data.get('test', {})
                    rse = test_data.get('rse') or test_data.get('rrse')
                    mse = test_data.get('mse')
                    mae = test_data.get('mae')
                    corr = test_data.get('corr')
                    
                    row = [model, cluster, dataset, seed, rse, mse, mae, corr]
                    if config_keys:
                        row += [config_data.get(key) for key in config_keys]
                    results.append(row)
            except (KeyError, json.JSONDecodeError) as e:
                print(f"Error processing {res_path}: {e}")
                continue

    # Write to CSV
    csv_path = '/home/hwkang/CSE/CSE/test/aggregated_results.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['model', 'cluster', 'dataset', 'seed', 'rse', 'mse', 'mae', 'corr']
        if config_keys:
            header += config_keys
        writer.writerow(header)
        writer.writerows(results)
    
    print(f"Aggregated results saved to {csv_path}")

if __name__ == "__main__":
    aggregate_results()