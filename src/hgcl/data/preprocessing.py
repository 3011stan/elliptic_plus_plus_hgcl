"""Fit immutable preprocessing exclusively on caller-supplied training rows."""
from dataclasses import dataclass
import numpy as np
from hgcl.config import digest
from hgcl.data.schema import TX_COLUMNS,NATIVE_COLUMNS,ADDRESS_COLUMNS


@dataclass(frozen=True)
class Preprocessor:
    kind: str
    names: tuple
    raw_names: tuple
    median: tuple
    mean: tuple
    scale: tuple
    numeric_indices: tuple
    fit_ids_hash: str

    def to_dict(self):
        return {k:list(v) if isinstance(v,tuple) else v for k,v in self.__dict__.items()}

    @staticmethod
    def fit(frame, *, kind):
        raw_names={'transaction':TX_COLUMNS+['graph_num_input_addresses','graph_num_output_addresses'],
                   'principal':ADDRESS_COLUMNS,'native_wallet':NATIVE_COLUMNS}[kind]
        names=(TX_COLUMNS+[c+'__missing' for c in TX_COLUMNS]+raw_names[-2:] if kind=='transaction'
               else NATIVE_COLUMNS+[c+'__missing' for c in NATIVE_COLUMNS] if kind=='native_wallet' else raw_names)
        arr=frame.select(raw_names).to_numpy().astype(np.float64)
        if not len(arr):raise ValueError('No training rows for preprocessing')
        for i,c in enumerate(raw_names):
            if np.isnan(arr[:,i]).all():raise ValueError(f'Entirely missing training column: {c}')
        median=np.nanmedian(arr,axis=0)
        indices=tuple(i for i,c in enumerate(names) if not c.endswith('__missing') and not c.endswith('__observed_fraction'))
        keys=['tx_id','step'] if kind=='transaction' else ['address','step']
        pre=Preprocessor(kind,tuple(names),tuple(raw_names),tuple(median),tuple(np.zeros(len(names))),tuple(np.ones(len(names))),indices,
                         digest(frame.select(keys).sort(keys).rows()))
        x,_=pre.transform(frame)
        mean=np.zeros(len(names));scale=np.ones(len(names))
        mean[list(indices)]=x[:,list(indices)].mean(axis=0)
        scale[list(indices)]=x[:,list(indices)].std(axis=0)
        scale[scale==0]=1
        return Preprocessor(kind,tuple(names),tuple(raw_names),tuple(median),tuple(mean),tuple(scale),indices,pre.fit_ids_hash)

    def transform(self,frame):
        arr=frame.select(self.raw_names).to_numpy().astype(np.float64)
        if np.isinf(arr).any() or (arr<0).any():raise ValueError('Invalid raw preprocessing values')
        missing=np.isnan(arr)
        values=np.where(missing,np.asarray(self.median),arr)
        if self.kind=='transaction':values=np.column_stack([values[:,:15],missing[:,:15],values[:,15:]])
        elif self.kind=='native_wallet':values=np.column_stack([values,missing])
        values[:,list(self.numeric_indices)]=np.log1p(values[:,list(self.numeric_indices)])
        scaled=(values-np.asarray(self.mean))/np.asarray(self.scale)
        if not np.isfinite(values).all() or not np.isfinite(scaled).all():raise ValueError('Nonfinite transformed values')
        return values.astype(np.float32),scaled.astype(np.float32)
