#!/usr/bin/env python3
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("MEDICINAL_SEED_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA_ROOT = Path(os.environ.get("MEDICINAL_SEED_DATA_ROOT", ROOT / "data")).resolve()
SRC = Path(os.environ.get("MEDICINAL_SEED_SOURCE_DATA_ROOT", ROOT / "source_data")).resolve()
OUT = Path(
    os.environ.get(
        "MEDICINAL_SEED_VALIDATION_OUTPUT",
        ROOT / "validation" / "validation_results.csv",
    )
).resolve()
checks = []
def add(name, expected, observed, tol=0):
    diff = observed - expected if isinstance(expected, (int,float)) else ""
    ok = abs(diff) <= tol if isinstance(diff, (int,float)) else expected == observed
    checks.append({"check":name,"expected":expected,"observed":observed,"difference":diff,"status":"PASS" if ok else "FAIL"})

train=pd.read_csv(SRC/'stage1/train_split.csv'); test=pd.read_csv(SRC/'stage1/test_split.csv')
pred=pd.read_csv(SRC/'stage1/convnext_small_test_predictions.csv')
assoc=pd.read_csv(SRC/'Supplementary_Table_S2_zero_baseline_absolute_IG_descriptor_association.csv')
inv=pd.read_csv(SRC/'stage7c/descriptor_map_inventory.csv')
corr=pd.read_csv(DATA_ROOT/'metadata/manual_orientation_corrections.csv')
add('training sample count',899,len(train)); add('test sample count',225,len(test))
add('descriptor count',29,len(inv)); add('ConvNeXt correct predictions',224,int((pred.true_label==pred.pred_label).sum()))
add('ConvNeXt accuracy',224/225,float((pred.true_label==pred.pred_label).mean()),1e-12)
add('unique manually corrected images',24,int(corr.stem.nunique()))
add('initial horizontal flips',15,int((corr.initial_operation=='hflip').sum()))
add('initial vertical flips',4,int((corr.initial_operation=='vflip').sum()))
add('initial rotation corrections',5,int(corr.initial_operation.str.startswith('rot').sum()))
for key,expected in [('LAB_L',0.4856638137944986),('Brightness',0.4848574373139294),('FFT_LowPass',0.4803676099889092)]:
    obs=float(assoc.loc[assoc['Descriptor key']==key,'Mean Spearman r'].iloc[0]); add(key+' mean Spearman r',expected,obs,1e-12)
out=pd.DataFrame(checks); OUT.parent.mkdir(parents=True, exist_ok=True); out.to_csv(OUT,index=False)
print(out.to_string(index=False)); raise SystemExit(1 if (out.status=='FAIL').any() else 0)
