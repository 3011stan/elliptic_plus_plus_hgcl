import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser(description='Audit original Elliptic++ CSVs without loading full feature matrices.')
parser.add_argument('--data-root', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
source = args.data_root.resolve()
output = args.output_dir.resolve()
if not source.is_dir():
    parser.error('data-root must be an existing directory')
if output == source or source in output.parents:
    parser.error('output-dir must be outside the original dataset directory')
output.mkdir(parents=True, exist_ok=True)
report = {'inspected_at': datetime.now(timezone.utc).isoformat(), 'source_at_inspection': str(source), 'files': {}, 'relations': {}}
wallet_times, tx_times, labels, first_features = {}, {}, {}, {}
changed_wallets = set()

order = ['wallets_classes.csv', 'wallets_features.csv', 'txs_classes.csv', 'txs_features.csv', 'wallets_features_classes_combined.csv', 'AddrTx_edgelist.csv', 'TxAddr_edgelist.csv', 'AddrAddr_edgelist.csv', 'txs_edgelist.csv']
for name in order:
    p = source / name
    with p.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    entry = {'bytes': p.stat().st_size, 'sha256': digest}
    rows, widths, times, classes, anomalies = 0, Counter(), Counter(), Counter(), Counter()
    seen = set()
    with p.open(newline='') as stream:
        reader = csv.reader(stream)
        header = next(reader)
        entry['columns'] = header
        for row in reader:
            rows += 1
            widths[len(row)] += 1
            if len(row) != len(header):
                anomalies['invalid_width'] += 1
                continue
            if name == 'wallets_classes.csv':
                if row[0] in labels:
                    anomalies['duplicate_id'] += 1
                    if labels[row[0]] != row[1]:
                        anomalies['conflicting_label'] += 1
                labels[row[0]] = row[1]
                classes[row[1]] += 1
            elif name == 'wallets_features.csv':
                key, t = row[0], int(row[1])
                bit = 1 << t
                prev = wallet_times.get(key, 0)
                anomalies['duplicate_address_time'] += bool(prev & bit)
                wallet_times[key] = prev | bit
                times[t] += 1
                anomalies['missing_label_id'] += key not in labels
                value_hash = hashlib.sha256('\x1f'.join(row[2:]).encode()).digest()
                if key in first_features and first_features[key] != value_hash:
                    changed_wallets.add(key)
                first_features.setdefault(key, value_hash)
            elif name == 'txs_features.csv':
                key, t = row[0], int(row[1])
                anomalies['duplicate_id'] += key in tx_times
                tx_times[key] = t
                times[t] += 1
            elif name == 'txs_classes.csv':
                anomalies['duplicate_id'] += row[0] in seen
                seen.add(row[0])
                classes[row[1]] += 1
            elif name == 'wallets_features_classes_combined.csv':
                key, t, label = row[:3]
                times[int(t)] += 1
                classes[label] += 1
                anomalies['label_mismatch'] += labels.get(key) != label
                anomalies['missing_feature_key'] += not bool(wallet_times.get(key, 0) & (1 << int(t)))
                anomalies['features_differ_from_first_address_row'] += first_features.get(key) != hashlib.sha256('\x1f'.join(row[3:]).encode()).digest()
            elif name in ('AddrTx_edgelist.csv', 'TxAddr_edgelist.csv'):
                wallet, tx = row if name == 'AddrTx_edgelist.csv' else (row[1], row[0])
                anomalies['missing_wallet'] += wallet not in wallet_times
                anomalies['missing_tx'] += tx not in tx_times
                t = tx_times.get(tx)
                if t is not None:
                    times[t] += 1
                    anomalies['wallet_not_observed_at_tx_time'] += not bool(wallet_times.get(wallet, 0) & (1 << t))
            elif name == 'AddrAddr_edgelist.csv':
                anomalies['missing_source_wallet'] += row[0] not in wallet_times
                anomalies['missing_target_wallet'] += row[1] not in wallet_times
            elif name == 'txs_edgelist.csv':
                anomalies['missing_source_tx'] += row[0] not in tx_times
                anomalies['missing_target_tx'] += row[1] not in tx_times
                if row[0] in tx_times and row[1] in tx_times:
                    anomalies['cross_time_edge'] += tx_times[row[0]] != tx_times[row[1]]
    entry.update({'data_rows': rows, 'column_count': len(header), 'row_width_counts': dict(widths), 'time_counts': dict(sorted(times.items())), 'class_counts': dict(classes), 'checks': dict(anomalies)})
    report['files'][name] = entry
    print(json.dumps({'file': name, 'rows': rows, 'columns': len(header), 'checks': dict(anomalies)}), flush=True)
    (output / 'initial-audit.json').write_text(json.dumps(report, indent=2) + '\n')

report['wallet_identity'] = {'unique_addresses': len(wallet_times), 'addresses_in_multiple_timesteps': sum(bits.bit_count() > 1 for bits in wallet_times.values()), 'addresses_with_different_non_time_features_across_rows': len(changed_wallets), 'unique_address_time_pairs': sum(bits.bit_count() for bits in wallet_times.values())}
report['transaction_identity'] = {'unique_tx_ids': len(tx_times)}
report['limitations'] = ['No exhaustive numeric validity or missing-value scan.', 'No duplicate-edge count.', 'Feature availability at historical prediction time has not been established.', 'Hashes identify supplied files; they do not authenticate an upstream release.']
(output / 'initial-audit.json').write_text(json.dumps(report, indent=2) + '\n')
manifest = {'source_at_inspection': str(source), 'files': {name: {'bytes': entry['bytes'], 'sha256': entry['sha256']} for name, entry in report['files'].items()}}
(output / 'source-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'wallet_identity': report['wallet_identity'], 'transaction_identity': report['transaction_identity']}), flush=True)
