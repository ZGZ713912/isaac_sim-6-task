#!/usr/bin/env python3
"""Export audited raw sweep states using the prior RMCS preprocessing pipeline.

Run in the NumPy/Pandas analysis environment, not the Isaac environment.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rmcs",type=Path,required=True)
    p.add_argument("--old-csv",type=Path,required=True)
    p.add_argument("--pp-csv",type=Path,required=True)
    p.add_argument("--cache-dir",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    sys.path.insert(0,str(a.rmcs/".script/identification"))
    from fit_leg_gas_spring import load_source, prepare
    from analyze_wheel_leg_sweep_archive import sha256
    arrays=[];sources=[]
    for number,path,stages in [(0,a.old_csv,range(1,13)),(1,a.pp_csv,range(1,12))]:
        frame,meta=load_source(path,a.cache_dir)
        x=prepare(frame,stages)
        x["source"]=number
        arrays.append(x);sources.append(meta)
    a.out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out/"real_data.npz",**{col:np.r_[arrays[0][col],arrays[1][col]] for col in arrays[0].columns})
    (a.out/"real_data.json").write_text(json.dumps(dict(sources=sources,
        preprocessing_source=str(a.rmcs/".script/identification/fit_leg_gas_spring.py"),
        preprocessing_sha256=sha256(a.rmcs/".script/identification/fit_leg_gas_spring.py"),
        interpretation="DM reported torque in model coordinates; no independent force sensor",
        conditions="user confirms horizontal suspended, zero gravity/external load"),indent=2,ensure_ascii=False)+"\n")
    print("EXPORTED",sum(len(x) for x in arrays),flush=True)


if __name__=="__main__":main()
