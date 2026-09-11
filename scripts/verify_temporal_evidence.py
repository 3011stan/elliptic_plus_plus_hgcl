"""Independently verify INV-001 selected examples from original rows using SQLite.

Uses SQL DISTINCT counts rather than the main investigation's bitmasks/dictionaries.
Validates the selected sample and total-count exceptions, not the full aggregate scan.
"""
import argparse
import csv
import hashlib
import json
import sqlite3
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--evidence-dir', type=Path, required=True)
    args = parser.parse_args()
    data, evidence = args.data_root.resolve(), args.evidence_dir.resolve()
    if evidence == data or data in evidence.parents or evidence in data.parents:
        parser.error('Evidence directory must be separate from original data')
    report = json.loads((evidence/'report.json').read_text())
    for name, identity in report['inputs'].items():
        if sha(data/name) != identity['sha256']:
            raise ValueError('Original input hash mismatch: '+name)
    for name, identity in report['artifacts'].items():
        if sha(evidence/name) != identity['sha256']:
            raise ValueError('Evidence artifact hash mismatch: '+name)
    sample = set(report['selected_addresses']) | {x['address'] for x in report['total_txs_exceptions']}
    db = sqlite3.connect(':memory:')
    db.executescript('CREATE TABLE wallet(address TEXT, t INTEGER, steps INTEGER, total INTEGER, last INTEGER); CREATE TABLE edge(address TEXT, tx TEXT); CREATE TABLE tx(tx TEXT PRIMARY KEY, t INTEGER);')
    original_examples = {}
    with (data/'wallets_features.csv').open(newline='') as stream:
        for line, row in enumerate(csv.DictReader(stream), 2):
            if row['address'] in sample:
                original_examples[line] = row
                db.execute('INSERT INTO wallet VALUES(?,?,?,?,?)', (row['address'], int(row['Time step']),
                    int(float(row['num_timesteps_appeared_in'])), int(float(row['total_txs'])), int(float(row['last_block_appeared_in']))))
    for name, address_col in [('AddrTx_edgelist.csv','input_address'),('TxAddr_edgelist.csv','output_address')]:
        with (data/name).open(newline='') as stream:
            for row in csv.DictReader(stream):
                if row[address_col] in sample:
                    db.execute('INSERT INTO edge VALUES(?,?)',(row[address_col],row['txId']))
    wanted_tx = {row[0] for row in db.execute('SELECT DISTINCT tx FROM edge')}
    with (data/'txs_features.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['txId'] in wanted_tx:
                db.execute('INSERT INTO tx VALUES(?,?)',(row['txId'],int(row['Time step'])))
    checked_pairs = checked_rows = 0
    with (evidence/'examples.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            address, t = row['address'], int(row['time_step'])
            actual = db.execute('SELECT COUNT(DISTINCT t),MIN(steps),MIN(total),MIN(last) FROM wallet WHERE address=? AND t<=?',(address,t)).fetchone()
            expected = tuple(int(row[c]) for c in ('observed_steps_through_t','num_timesteps_appeared_in','total_txs','last_block_appeared_in'))
            assert actual == expected, (address,t,actual,expected)
            for restriction, column in [('tx.t <= ?', 'unique_dataset_transactions_through_t'),('tx.t = ?', 'unique_dataset_transactions_at_t')]:
                count = db.execute('SELECT COUNT(DISTINCT edge.tx) FROM edge JOIN tx USING(tx) WHERE address=? AND '+restriction,(address,t)).fetchone()[0]
                assert count == int(row[column]), (address,t,column,count)
            count = db.execute('SELECT COUNT(DISTINCT tx) FROM edge WHERE address=?',(address,)).fetchone()[0]
            assert count == int(row['unique_dataset_transactions_full'])
            checked_pairs += 1
    with (evidence/'example-source-rows.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            source = original_examples[int(row['source_line'])]
            assert all(source[k] == v for k,v in row.items() if k != 'source_line')
            checked_rows += 1
    for item in report['total_txs_exceptions']:
        address = item['address']
        actual = db.execute('SELECT COUNT(DISTINCT tx) FROM edge WHERE address=?',(address,)).fetchone()[0]
        published = db.execute('SELECT DISTINCT total FROM wallet WHERE address=?',(address,)).fetchall()
        assert actual == item['distinct_dataset_transactions']
        assert published == [(item['published_total_txs'],)]
    result = {'status':'passed', 'method':'independent raw CSV reread + SQLite DISTINCT joins',
              'scope':'selected examples and recorded total_txs exceptions; aggregate scan not independently repeated',
              'address_time_examples_checked':checked_pairs, 'source_rows_checked':checked_rows,
              'exceptions_checked':len(report['total_txs_exceptions']), 'label_files_read':[],
              'report_sha256':sha(evidence/'report.json'), 'verification_script_sha256':sha(Path(__file__))}
    (evidence/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
