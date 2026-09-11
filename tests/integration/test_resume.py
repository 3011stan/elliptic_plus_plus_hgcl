"""Replay the actual SSL loop, not only optimizer serialization."""
from pathlib import Path
import json
import numpy as np
import torch
import pytest
from hgcl.config import defaults
from hgcl.training.pretrain import pretrain
from hgcl.provenance import file_hash


def small_prepared(root):
    random=np.random.default_rng(11)
    for step in [1,2]:
        folder=root/'principal'/f'step-{step:02}';folder.mkdir(parents=True)
        arrays={'address_x':random.normal(size=(8,6)).astype('float32'),'transaction_x':random.normal(size=(4,4)).astype('float32'),
                'address_ids':np.array([f'a{i}' for i in range(8)]),'transaction_ids':np.array([f't{i}' for i in range(4)]),'tabular_x':np.zeros((8,6),dtype='float32')}
        arrays['sender']=np.array([np.arange(8),np.arange(8)%4],dtype='int64');arrays['receiver']=np.array([np.arange(4),np.arange(4)],dtype='int64')
        for r in ('sender','receiver'):arrays[r+'_reverse']=arrays[r][::-1].copy()
        for name,a in arrays.items():np.save(folder/(name+'.npy'),a,allow_pickle=False)
    (root/'principal/address-preprocessor.json').write_text(json.dumps({'numeric_indices':list(range(6))}))
    (root/'transaction-preprocessor.json').write_text(json.dumps({'numeric_indices':list(range(4))}))


def test_ssl_epoch_replay_and_immutable_reuse(tmp_path):
    prepared=tmp_path/'prepared';small_prepared(prepared);v=defaults('smoke');v['encoder']['hidden']=8;v['ssl']['projection_dim']=8
    expected,report=pretrain(prepared,'principal',v,11,tmp_path/'reference',{'test':'fixture'},lambda:None)
    with pytest.raises(InterruptedError):pretrain(prepared,'principal',v,11,tmp_path/'resumed',{'test':'fixture'},lambda:None,stop_after_epoch=1)
    actual,replayed=pretrain(prepared,'principal',v,11,tmp_path/'resumed',{'test':'fixture'},lambda:None)
    for k in expected:torch.testing.assert_close(expected[k],actual[k],rtol=0,atol=0)
    assert replayed==report and report['encoder_changed']
    path=tmp_path/'resumed/ssl-final.pt';sha=file_hash(path)
    again,_=pretrain(prepared,'principal',v,11,tmp_path/'resumed',{'test':'fixture'},lambda:None)
    assert sha==file_hash(path)
    for k in expected:torch.testing.assert_close(expected[k],again[k],rtol=0,atol=0)
