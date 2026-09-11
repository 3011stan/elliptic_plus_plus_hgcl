"""INV-001: label-free temporal feature checks using only the Python standard library.

Reads four original CSVs, verifies their manifest hashes before/after analysis, and
writes derived evidence outside the data directory. No model is trained.
"""

import argparse
import csv
import hashlib
import json
import platform
import resource
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


COLUMNS = (
    'num_timesteps_appeared_in', 'total_txs', 'last_block_appeared_in',
    'first_block_appeared_in', 'num_txs_as_sender', 'num_txs_as receiver',
)
FILES = ('wallets_features.csv', 'txs_features.csv',
         'AddrTx_edgelist.csv', 'TxAddr_edgelist.csv')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def steps(mask):
    return [t for t in range(1, 50) if mask & (1 << t)]


def integer(value):
    x = float(value)
    if not x.is_integer() or x < 0:
        raise ValueError(f'Expected nonnegative integral count/block: {value!r}')
    return int(x)


def run(data, output, manifest_path):
    started = time.perf_counter()
    manifest = json.loads(manifest_path.read_text())['files']
    identities = {}
    for name in FILES:
        p = data / name
        sha = digest(p)
        if sha != manifest[name]['sha256']:
            raise ValueError(f'Manifest hash mismatch: {name}')
        identities[name] = {'bytes': p.stat().st_size, 'sha256': sha}
    print('Input hashes verified; reading wallet features.', flush=True)
    wallets, checks = {}, Counter()
    with (data / FILES[0]).open(newline='') as stream:
        reader = csv.reader(stream)
        header = next(reader)
        indices = [header.index(c) for c in COLUMNS]
        for line, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f'Invalid wallet row width at line {line}')
            address, t = row[0], integer(row[1])
            if not 1 <= t <= 49:
                raise ValueError(f'Unexpected time step: {t}')
            values = tuple(integer(row[i]) for i in indices)
            record = wallets.get(address)
            if record is None:
                wallets[address] = [1 << t, values, 1]
            else:
                if values != record[1]:
                    raise ValueError(f'Investigated attributes vary for {address}')
                checks['extra_address_time_rows'] += bool(record[0] & (1 << t))
                record[0] |= 1 << t
                record[2] += 1
            checks['wallet_rows'] += 1
    recurrent = [a for a, r in wallets.items() if r[0].bit_count() > 1]
    # Fixed, label-free selection: hash-ranked recurrent addresses plus lexical anchors.
    selected = sorted(set(sorted(recurrent, key=lambda a: hashlib.sha256(a.encode()).hexdigest())[:10]
                          + sorted(recurrent)[:3]))
    tx_times, tx_headers = {}, []
    with (data / FILES[1]).open(newline='') as stream:
        reader = csv.reader(stream)
        tx_headers = next(reader)
        for row in reader:
            if len(row) != len(tx_headers):
                raise ValueError('Invalid transaction row width')
            tx, t = integer(row[0]), integer(row[1])
            if tx in tx_times or not 1 <= t <= 49:
                raise ValueError('Invalid transaction identity/time')
            tx_times[tx] = t
    print('Wallet and transaction tables read; joining deduplicated incidences.', flush=True)
    incidents = {}
    sample_edges = []
    sample_set = set(selected)
    for name in FILES[2:]:
        with (data / name).open(newline='') as stream:
            reader = csv.reader(stream)
            next(reader)
            for line, row in enumerate(reader, 2):
                if len(row) != 2:
                    raise ValueError(f'Invalid edge width in {name}:{line}')
                address, tx = row if name == FILES[2] else (row[1], row[0])
                tx = integer(tx)
                if address not in wallets or tx not in tx_times:
                    raise ValueError('Missing edge endpoint')
                t = tx_times[tx]
                if not wallets[address][0] & (1 << t):
                    raise ValueError('Edge step missing from wallet observations')
                # One transaction counts once even when the address sends and receives.
                incident = incidents.setdefault(address, {})
                bit = 1 if name == FILES[2] else 2
                prior = incident.get(tx, 0)
                checks[name + '_duplicate_address_tx_rows'] += bool(prior & bit)
                incident[tx] = prior | bit
                checks[name + '_rows'] += 1
                if address in sample_set:
                    sample_edges.append({'address': address, 'txId': tx, 'time_step': t,
                                         'relation': name, 'source_line': line})
    comparisons = Counter()
    total_exceptions = []
    earliest_step_examples = []
    for address, (mask, values, rows) in wallets.items():
        observed = steps(mask)
        published_steps, total, last, first, sender, receiver = values
        events = incidents.get(address, {})
        event_times = Counter(tx_times[tx] for tx in events)
        if set(event_times) != set(observed):
            raise ValueError('Wallet step set differs from incident transaction step set')
        comparisons['num_steps_equal_full_observed'] += published_steps == len(observed)
        comparisons['num_steps_not_equal_full_observed'] += published_steps != len(observed)
        comparisons['published_steps_exceed_prefix_pairs'] += sum(published_steps > k for k in range(1, len(observed)+1))
        comparisons['total_equal_unique_dataset_transactions'] += total == len(events)
        comparisons['total_greater_than_unique_dataset_transactions'] += total > len(events)
        comparisons['total_less_than_unique_dataset_transactions'] += total < len(events)
        if total != len(events):
            total_exceptions.append({'address': address, 'steps': observed,
                                     'published_total_txs': total,
                                     'distinct_dataset_transactions': len(events),
                                     'num_txs_as_sender': sender,
                                     'num_txs_as_receiver': receiver,
                                     'transaction_ids': sorted(events)})
        comparisons['total_equal_sender_plus_receiver_fields'] += total == sender + receiver
        comparisons['sender_equal_distinct_sent_edges'] += sender == sum(bool(flags & 1) for flags in events.values())
        comparisons['receiver_equal_distinct_received_edges'] += receiver == sum(bool(flags & 2) for flags in events.values())
        comparisons['last_block_before_first_block'] += last < first
        comparisons['recurrent_equal_first_last_block'] += len(observed) > 1 and first == last
        comparisons['addresses_cross_candidate_34_35_cut'] += observed[0] <= 34 < observed[-1]
        if len(observed) > 1:
            comparisons['recurrent_total_equal_unique_dataset_transactions'] += total == len(events)
            comparisons['recurrent_total_exceeds_first_step_observed'] += total > event_times[observed[0]]
        cumulative_count = 0
        for t in observed:
            cumulative_count += event_times[t]
            comparisons['full_dataset_total_matches_but_exceeds_prefix_pairs'] += total == len(events) and total > cumulative_count
        if address in sample_set:
            cumulative = 0
            for rank, t in enumerate(observed, 1):
                cumulative += event_times[t]
                earliest_step_examples.append({
                    'address': address, 'time_step': t,
                    **dict(zip(COLUMNS, values)),
                    'observed_steps_through_t': rank,
                    'observed_steps_full': len(observed),
                    'unique_dataset_transactions_at_t': event_times[t],
                    'unique_dataset_transactions_through_t': cumulative,
                    'unique_dataset_transactions_full': len(events),
                    'first_observed_step': observed[0], 'last_observed_step': observed[-1],
                })
    # Direct source row locators for all selected address-time occurrences.
    sample_rows = []
    with (data / FILES[0]).open(newline='') as stream:
        reader = csv.DictReader(stream)
        for line, row in enumerate(reader, 2):
            if row['address'] in sample_set:
                sample_rows.append({'source_line': line, 'address': row['address'],
                                    'Time step': row['Time step'],
                                    **{c: row[c] for c in COLUMNS}})
    print('Checking that original files are unchanged.', flush=True)
    for name in FILES:
        if digest(data / name) != identities[name]['sha256']:
            raise ValueError(f'Input changed during analysis: {name}')
    output.mkdir(parents=True, exist_ok=True)
    tables = {'examples.csv': earliest_step_examples,
              'example-source-rows.csv': sample_rows,
              'example-edges.csv': sorted(sample_edges, key=lambda r: (r['address'], r['time_step'], r['txId'], r['relation']))}
    for name, rows in tables.items():
        with (output / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = {
        'investigation': 'INV-001', 'completed_at': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'data_root': str(data),
        'script_sha256': digest(Path(__file__)), 'manifest_sha256': digest(manifest_path),
        'inputs': identities, 'input_hashes_unchanged_after_analysis': True,
        'label_files_read': [], 'labels_used_for_selection': False,
        'wallet_feature_columns': header, 'transaction_feature_columns': tx_headers,
        'checks': dict(checks), 'addresses': len(wallets), 'recurrent_addresses': len(recurrent),
        'distinct_address_time_pairs': sum(r[0].bit_count() for r in wallets.values()),
        'transactions': len(tx_times), 'comparisons': dict(comparisons),
        'total_txs_exceptions': total_exceptions,
        'selection_rule': '10 smallest SHA256(address UTF-8) among recurrent addresses plus 3 lexicographically smallest recurrent addresses; sorted union',
        'selected_addresses': selected,
        'example_table_rows': len(earliest_step_examples),
        'block_time_mapping': 'No explicit timestamp/block-height column in txs_features.csv; no verified mapping established by this script.',
        'elapsed_seconds': round(time.perf_counter() - started, 3),
        'peak_rss_platform_units': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'artifacts': {name: {'sha256': digest(output/name), 'rows': len(rows)} for name, rows in tables.items()},
    }
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('addresses', 'recurrent_addresses', 'comparisons', 'elapsed_seconds', 'peak_rss_platform_units')}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1]/'docs/data/source-manifest.json')
    args = parser.parse_args()
    data, output = args.data_root.resolve(), args.output_dir.resolve()
    if output == data or data in output.parents or output in data.parents:
        parser.error('Output must be separate from the original data directory and its ancestors')
    run(data, output, args.manifest.resolve())


if __name__ == '__main__':
    main()
