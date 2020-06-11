"""
This script generates a 3-column data file with patient id, offset of first
medication, and offset of first treatment, left blank if there wasn't any.
"""

import argparse
import os
import pandas as pd

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


def get_patients(data_supdir):
    pts = [os.path.splitext(f)[0] for f in os.listdir( pj(data_supdir,
        "async_data") )]
    return pts

def main(cfg):
    data_supdir = cfg["data_supdir"]
    pts = get_patients(data_supdir)
    rows = []
    for pt in pts:
        row = [pt,]
        df_meds = pd.read_csv( pj(data_supdir, "async_meds", pt+".csv") )
        starts = df_meds[ df_meds["event"]=="drugstart" ]["eventoffset"]
        if len(starts) == 0:
            row.append(None)
        else:
            row.append( int( starts.iloc[0] ) )

        df_treat = pd.read_csv( pj(data_supdir, "async_treat", pt+".csv") )
        offsets = df_treat["treatmentoffset"]
        if len(offsets) == 0:
            row.append(None)
        else:
            row.append( int( offsets.iloc[0] ) )

        rows.append(row)

    df = pd.DataFrame(rows, columns=["patient", "first_med", "first_treat"])
    df.to_csv( pj(data_supdir, "pt_firsts.csv") )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-supdir", type=str,
            default=pj(HOME, "Datasets/EHRs/PhysioNet/eICU"))
    cfg = vars( parser.parse_args() )
    main(cfg)

