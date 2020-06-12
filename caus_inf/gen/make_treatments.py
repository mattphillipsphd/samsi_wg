"""
Generate the asynchronous treatment data files
"""

import argparse
import csv
import os
import pandas as pd

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


def get_patient_list(cfg):
    pt_files = os.listdir( pj(cfg["data_supdir"], "async_data") )
    pts = [int( os.path.splitext(f)[0] ) for f in pt_files]
    return pts

def main(cfg):
    data_supdir = cfg["data_supdir"]
    output_dir = pj(data_supdir, "async_treat")
    pts = get_patient_list(cfg)
    treats_df = pd.read_csv( pj(data_supdir, "csv/treatment.csv") )
    treat_vars = ["treatmentid", "treatmentoffset", "treatmentstring",
            "activeupondischarge"]
    for pt in pts:
        treats_p = treats_df[ treats_df["patientunitstayid"]==pt ]
        treats_p = treats_p[treat_vars] 
        treats_p = treats_p.sort_values("treatmentoffset")
        treats_p.to_csv( pj(output_dir, "%d.csv" % pt) )        
    
    output_dir = pj(data_supdir, "async_meds")
    med_events = ["drugstartoffset", "drugstopoffset"]
    med_vars = ["medicationid", "drugname", "drugivadmixture",
            "drugordercancelled", "drugname", "drughiclseqno", "dosage",
            "routeadmin", "frequency", "loadingdose", "prn", "gtc"]
    meds_df = pd.read_csv( pj(data_supdir, "csv/medication.csv") )
    for pt in pts:
        meds_p_orig = meds_df[ meds_df["patientunitstayid"]==pt ]
        meds_p = meds_p_orig[med_vars] 
        meds_p.insert(0, "event", "")
        ev_dfs = []
        for ev in med_events:
            ev_df = meds_p.copy()
            ev_df.insert(0, "eventoffset", meds_p_orig[ev])
            ev_df.loc[:,"event"] = ev[:-6] # Remove trailing 'offset'
            ev_dfs.append(ev_df)
        meds_p = pd.concat(ev_dfs, ignore_index=True)
        meds_p = meds_p.sort_values("eventoffset")
        meds_p.to_csv( pj(output_dir, "%d.csv" % pt) )        
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-supdir", type=str,
            default=pj(HOME, "Datasets/EHRs/PhysioNet/eICU"))
    cfg = vars( parser.parse_args() )
    main(cfg)

