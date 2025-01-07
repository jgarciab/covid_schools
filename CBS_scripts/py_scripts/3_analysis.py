#!/usr/bin/env python
# coding: utf-8


#Imports
import os
import pickle


import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns

from common_functions import *
from collections import Counter
from statsmodels.stats.proportion import proportion_confint

import dask
from dask.dataframe import read_csv
from time import time
from scipy import sparse


pd.options.display.max_columns = 100
pd.options.mode.chained_assignment = None

data_path = "H:/data_overload/network_creation/data/" 
temp_data = "T:/"

import numpy as np
from statsmodels.stats.proportion import proportion_confint

def calc_prop(df):
    """
    Calculates the confidence interval for the proportion of co-infected individuals.

    This function computes a 95% confidence interval for the proportion of co-infected 
    individuals in a given dataset and returns the result as a percentage.

    Args:
        df (pd.DataFrame): A DataFrame containing a column named `co_infected`, 
                           where each entry is a binary value (0 or 1) indicating 
                           whether an individual is co-infected.

    Returns:
        np.ndarray: A NumPy array with the lower and upper bounds of the 95% confidence interval, 
                    expressed as percentages.

    Notes:
        Assumes an infinite population
    """
    # Calculate the sum of co-infected cases and the total number of records
    co_infected_sum = df['co_infected'].sum()
    total_count = len(df)

    # Calculate the 95% confidence interval using Wilson method (default)
    conf_int = proportion_confint(co_infected_sum, total_count)

    # Convert the confidence interval to percentages
    return 100 * np.array(conf_int)



def save_calc_prop(df, schools=True, label="", data_path="data_path"):
    """
    Saves calculated proportions and statistics to a file for different distance thresholds and conditions.

    This function calculates and saves the proportion of co-infected individuals 
    for various distance thresholds and optionally for specific grouping conditions (e.g., within the same school or municipality). 
    Results are appended to a tab-separated values (TSV) file.

    Args:
        df (pd.DataFrame): A DataFrame containing columns `distance`, `co_infected`, 
                           `POSTCODE1`, `POSTCODE2`, `gemcode1`, and `gemcode2`.
        schools (bool, optional): If True, includes calculations for cases within the same school 
                                  (`POSTCODE1` == `POSTCODE2`) and municipality (`gemcode1` == `gemcode2`). Defaults to True.
        label (str, optional): A label to prepend to each row of output for context. Defaults to an empty string.
        data_path (str, optional): Path to the directory where the output file (`stats_full.tsv`) is saved. 
                                   Defaults to "data_path".

    Returns:
        None: The function appends results to a file and prints statistics to the console.

    Notes:
        - The `distance` column is used to group data into predefined thresholds.
        - The output file is named `stats_full.tsv` and is tab-separated.

    """
    # Open the output file in append mode
    with open(f"{data_path}/stats_full.tsv", "a+") as f:
        # General statistics for the entire dataset
        print(f"General prob (N={len(df)}) {calc_prop(df)} {df['co_infected'].sum()}")
        f.write(f"{label}\tgeneral\t{len(df)}\t{df['co_infected'].sum()}\n")
            
        # Threshold ranges for distance calculations
        th_p = -1
        for threshold in [0, 300, 1000, 3000, 10000, 30000, 300000]:
            # Filter rows based on the current distance threshold range
            d_th = df.loc[(df['distance'] > th_p) & (df['distance'] <= threshold)]
            print(f"Within ({th_p}-{threshold}] meters (N={len(d_th)}) {calc_prop(d_th)} {d_th['co_infected'].sum()}")
            f.write(f"{label}\t{th_p}-{threshold}\t{len(d_th)}\t{d_th['co_infected'].sum()}\n")
            th_p = threshold  # Update the previous threshold for the next iteration
        
        # Additional grouping calculations if `schools` is True
        if schools:
            # Calculate for individuals within the same postcode
            d_th = df.loc[df['POSTCODE1'] == df['POSTCODE2']]
            print(f"Within postcode (N={len(d_th)}) {calc_prop(d_th)} {d_th['co_infected'].sum()}")
            f.write(f"{label}\tschool_postcode\t{len(d_th)}\t{d_th['co_infected'].sum()}\n")

            # Calculate for individuals within the same municipality (gemeente)
            d_th = df.loc[df['gemcode1'] == df['gemcode2']]
            print(f"Within gemeente (N={len(d_th)}) {calc_prop(d_th)} {d_th['co_infected'].sum()}")
            f.write(f"{label}\tschool_gemeente\t{len(d_th)}\t{d_th['co_infected'].sum()}\n")

# Read RIMV data
rivm = read_rivm(only_positives=True)



# Read family network (using dask)
path_network = f"G:/Bevolking/PN/PersNw2018_v1.0_links_familie.csv"
df_jan_fam = read_csv(path_network, sep=";",dtype=str,usecols=["RINPERSOONSRC","RINPERSOONDST","linktype"])
df_jan_fam = df_jan_fam.loc[df_jan_fam["linktype"] == "103"].compute()
set_siblings = set(df_jan_fam["RINPERSOONSRC"]+df_jan_fam["RINPERSOONDST"])


# Read groups 1-4 (from script 2)
bo = pickle.load(open(f"{temp_data}/student_pairs.csv", "rb"))
baseline = pickle.load(open(f"{temp_data}/student_pairs_baseline.csv", "rb"))



# When the students live in the same house the distance should be 0
bo.loc[bo["RINOBJECTNUMMER"]==bo["RINOBJECTNUMMER2"], "distance"] = 0
baseline.loc[baseline["RINOBJECTNUMMER"]==baseline["RINOBJECTNUMMER2"], "distance"] = 0


# Add first infection date
bo["date_infection_1"] = bo["RINPERSOON1"].map(rivm)
bo["date_infection_2"] = bo["RINPERSOON2"].map(rivm)

baseline["date_infection_1"] = baseline["RINPERSOON1"].map(rivm)
baseline["date_infection_2"] = baseline["RINPERSOON2"].map(rivm)


# Temporally associated infections
threshold = 14
bo["co_infected"] = ((bo["date_infection_1"]-bo["date_infection_2"]).abs() < threshold)
baseline["co_infected"] = ((baseline["date_infection_1"]-baseline["date_infection_2"]).abs() < threshold)



# test the threshold
sns.distplot((baseline["date_infection_1"]-baseline["date_infection_2"]), hist=False)
sns.distplot((bo["date_infection_1"]-bo["date_infection_2"]),hist=False)
plt.plot([-7,-7],[0,0.004])
plt.plot([7,7],[0,0.004])


# Not infected
bo["not_infected"] = np.isnan(bo["date_infection_1"]) & np.isnan(bo["date_infection_2"]) 
baseline["not_infected"] = np.isnan(baseline["date_infection_1"]) & np.isnan(baseline["date_infection_2"]) 


# Add two ID columns to improve efficiency
bo["pair"] = bo["RINPERSOON1"] + bo["RINPERSOON2"]
baseline["pair"] = baseline["RINPERSOON1"] + baseline["RINPERSOON2"]


# Create samples (groups 1-4, note that the group numbers do not correspond to the paper)
# Same class vo
filter_ = ((bo["BRIN_crypt1"] == bo["BRIN_crypt2"]) & (bo["OPLNR"] == bo["OPLNR2"]) & (bo["BRINVEST1"] == bo["BRINVEST2"]))
print(filter_.sum())
type1_same_class = bo.loc[filter_]

# Different class vo
filter_ = ((bo["BRIN_crypt1"] == bo["BRIN_crypt2"]) & (bo["OPLNR"] != bo["OPLNR2"]) & (bo["BRINVEST1"] == bo["BRINVEST2"]))
print(filter_.sum())
type2_diff_class = bo.loc[filter_]

# Different school / same institution
filter_ = ((bo["BRIN_crypt1"] == bo["BRIN_crypt2"]) & (bo["BRINVEST1"] != bo["BRINVEST2"]))
print(filter_.sum())
type3_same_inst = bo.loc[filter_]


# Different school vo
filter_ = (bo["BRIN_crypt1"] != bo["BRIN_crypt2"])
print(filter_.sum())
type4_diff_school = bo.loc[filter_]

# Random sample (for comparison)
print(baseline.shape)


# Calculate proportions in general
print("Type 1, same class")
save_calc_prop(type1_same_class, label="same_class_all")

print("\nType 2, different class, same school")
save_calc_prop(type2_diff_class, label="same_school_all")

print("\nType 3, different school same institution")
save_calc_prop(type3_same_inst, label="same_institution_all")

print("\nType 4, different school")
save_calc_prop(type4_diff_school, label="different_inst_all")

print("\nType 4, baseline")
save_calc_prop(baseline, label="baseline_all")


# Calculate proportions  for twins
print("Type 1, same class")
save_calc_prop(type1_same_class.loc[type1_same_class["pair"].isin(set_siblings)], label="same_class_twins")

print("\nType 2, different class, same school")
save_calc_prop(type2_diff_class.loc[type2_diff_class["pair"].isin(set_siblings)],  label="same_school_twins")

print("\nType 3, different school same institution")
save_calc_prop(type3_same_inst.loc[type3_same_inst["pair"].isin(set_siblings)], label="same_institution_twins")

print("\nType 4, different school")
save_calc_prop(type4_diff_school.loc[type4_diff_school["pair"].isin(set_siblings)], label="different_inst_twins")

print("\nType 4, baseline")
save_calc_prop(baseline.loc[baseline["pair"].isin(set_siblings)], label="baseline_twins")

# Calculate proportions of infections in general 
print("Type 1, same class")
save_calc_prop(type1_same_class.loc[~type1_same_class["not_infected"]], label="same_class_infected")

print("\nType 2, different class, same school")
save_calc_prop(type2_diff_class.loc[~type2_diff_class["not_infected"]],  label="same_school_infected")

print("\nType 3, different school same institution")
save_calc_prop(type3_same_inst.loc[~type3_same_inst["not_infected"]], label="same_institution_infected")

print("\nType 4, different school")
save_calc_prop(type4_diff_school.loc[~type4_diff_school["not_infected"]], label="different_inst_infected")

print("\nType 4, baseline")
save_calc_prop(baseline.loc[~baseline["not_infected"]], label="baseline_infected")


# Calculate proportions of infections in general
print("Type 1, same class")
save_calc_prop(type1_same_class.loc[(~type1_same_class["not_infected"])&type1_same_class["pair"].isin(set_siblings)], label="same_class_infected_twins")

print("\nType 2, different class, same school")
save_calc_prop(type2_diff_class.loc[(~type2_diff_class["not_infected"])&type2_diff_class["pair"].isin(set_siblings)],  label="same_school_infected_twins")

print("\nType 3, different school same institution")
save_calc_prop(type3_same_inst.loc[(~type3_same_inst["not_infected"])&type3_same_inst["pair"].isin(set_siblings)], label="same_institution_infected_twins")

print("\nType 4, different school")
save_calc_prop(type4_diff_school.loc[(~type4_diff_school["not_infected"])&type4_diff_school["pair"].isin(set_siblings)], label="different_inst_infected_twins")

print("\nType 4, baseline")
save_calc_prop(baseline.loc[(~baseline["not_infected"])&baseline["pair"].isin(set_siblings)], label="baseline_infected_twins")


## CAlculate temporally associated infections for family


# Addresses of people
addressen = read_file_current_version("G:\Bevolking\GBAADRESOBJECTBUS", 2021, usecols=["RINPERSOON","GBADATUMAANVANGADRESHOUDING","GBADATUMEINDEADRESHOUDING","RINOBJECTNUMMER"])
# drop rows before 2021
addressen = addressen.loc[addressen["GBADATUMAANVANGADRESHOUDING"]<"20210000"]
addressen = addressen.loc[addressen["GBADATUMEINDEADRESHOUDING"]>"20210000"] #still living in the house
addressen = addressen.drop_duplicates(subset=["RINPERSOON"], keep="last")
# Merge addresses to coordinates (100x100 square)
coord = read_file_current_version("G:\BouwenWonen\VSLVIERKANTTAB", 2022, usecols=["RINOBJECTNUMMER", "VRLVIERKANT100M"]).drop_duplicates()

addressen = pd.merge(addressen[["RINPERSOON","RINOBJECTNUMMER"]], coord)
display(addressen.head(1))





## Read family
path_network = f"G:/Bevolking/PN/PersNw2018_v1.0_links_familie.csv"
df_jan_fam = read_csv(path_network, sep=";",dtype=str,usecols=["RINPERSOONSRC","RINPERSOONDST","linktype"])
df_jan_fam = df_jan_fam.loc[df_jan_fam["linktype"].isin({"102","103","104"})].compute()
    

# Calculate proportions for different type of family pairs
for label, code in zip(("Co-Parents", "Parent-child", "Siblings"), ("102", "104", "103")):
    print("\n\n", label)
    df = df_jan_fam.loc[df_jan_fam["linktype"]==code]
    df["date_infection_1"] = df["RINPERSOONSRC"].map(rivm)
    df["date_infection_2"] = df["RINPERSOONDST"].map(rivm)
    print(len(df))

    df = pd.merge(df, addressen.rename(columns={"RINPERSOON":"RINPERSOONSRC"}), on ="RINPERSOONSRC", validate="m:1")
    print(len(df))
    df = pd.merge(df, addressen.rename(columns={"RINPERSOON":"RINPERSOONDST"}), on ="RINPERSOONDST", suffixes=["","2"], validate="m:1")
    print(len(df))


    df["co_infected"] = ((df["date_infection_1"]-df["date_infection_2"]).abs() < threshold)
    df["not_infected"] = np.isnan(df["date_infection_1"]) & np.isnan(df["date_infection_2"]) 
    df = df.loc[(df["VRLVIERKANT100M"].str[0] != "-") & (df["VRLVIERKANT100M2"].str[0] != "-")]
    df = calculate_distance(df)
    df.loc[df["RINOBJECTNUMMER"]==df["RINOBJECTNUMMER2"], "distance"] = 0
    print(f"\n\n----------------------\nBetween {label} - all")
    save_calc_prop(df, schools=False, label=f"{label}-{code}_all")
    
    print(f"----------------------\nBetween {label} - only infected")
    save_calc_prop(df.loc[~df["not_infected"]], schools=False, label=f"{label}-{code}_infected")
    
    del df


## Save all results (to export)

df = pd.read_csv(f"{data_path}/stats_full.tsv", sep="\t", header=None)
df.columns = ["Group","Distance","N","N_inf"]
df.loc[df["N_inf"]<10, "N_inf"] = np.nan
df.loc[df["N"]<10, "N"] = np.nan
df.to_excel(f"{data_path}/stats.xlsx",index=None)




