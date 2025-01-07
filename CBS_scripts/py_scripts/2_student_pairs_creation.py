#!/usr/bin/env python
# coding: utf-8



#Imports
import os
import pickle


import numpy as np
import pandas as pd
import seaborn as sns

from common_functions import *

pd.options.display.max_columns = 100


data_path = "H:/data_overload/network_creation/data/" 
bipartite_data_path = "H:/data_overload/network_creation/data/bipartite" 
projected_data_path = "H:/data_overload/network_creation/data/projected"
temp_data = "T:/"


# #Read VO
# - A keep leerjaar 1 in 2020 (same school)
# - B keep leerlaar 1 in 2020 (different school, same gemeentes --> check brinaddressen)
# - C keep leerlaar 1 in 2020 (different gemeentes --> check brinaddressen)
# 
# #Read BO
# - keep leerjaar 8 in 2019
# 
# #Merge
# - Type 1: Same bo, same vo --> merge  with a
# - Type 2: Same bo, different vo closeby --> merge with b
# - Type 3: Same bo, different vo far apart --> merge with c
# - Type 4: Find a random sample of people not going to school together (ever), but in (a) same gemeente (b) different gemeente
# 

# Read addresses of the educational site in 2020
addressen = read_file_current_version("G:\Onderwijs\BRINADRESSEN", 2020)[["BRIN_crypt","BRINVest","RINObjectnummer","gemcode","POSTCODE","PLAATSNAAM"]]
addressen = addressen.rename(columns={"BRINVest":"BRINVEST"})

## GROUP 2-4: Students together in primary school

## Merge students who are in VO in 2020 
# Read secondary education (grade 9 only)
vo = pd.read_csv(f"{bipartite_data_path}/vo_2020.tsv", sep="\t", dtype=str)
vo = vo.loc[(vo["VOLEERJAAR"]=="leerjaar 1")]
vo = vo.rename(columns={"VOBRINVEST": "BRINVEST"})
vo["BRINVEST"] = vo["BRINVEST"].replace('geen codelijst beschikbaar, zie externe link',"00")
vo = vo.loc[~vo["RINPERSOON"].duplicated(keep=False)].dropna(subset=["RINPERSOON"]) #Remove students that went to two schools that year
print(vo.shape)
vo = pd.merge(vo, addressen, how="left", on=["BRIN_crypt","BRINVEST"], validate="m:1")
print(vo.shape)
print("VO cleaned")

# Read primary education (grade 8 only)
bo = pd.read_csv(f"{projected_data_path}/bo_2019_last_year.tsv", sep="\t", dtype=str) #only students in the last year studying 8 years in the same place
bo = bo.rename(columns={"WPOBRIN_crypt":"BRIN_crypt", "WPOBRINVEST": "BRINVEST"})
bo["BRINVEST"] = bo["BRINVEST"].replace('geen codelijst beschikbaar, zie externe link',"00")
print(bo.shape)
bo = pd.merge(bo, addressen, how="left", on=["BRIN_crypt","BRINVEST"],validate="m:1")
bo["RINPERSOON1"] = bo["RINPERSOON1"].str.strip()
bo["RINPERSOON2"] = bo["RINPERSOON2"].str.strip()

print(bo.shape)
print("BO cleaned")

print("\nStarting to merge, original shape")
print(bo.shape)
bo = pd.merge(bo, vo.rename(columns={"RINPERSOON":"RINPERSOON1"}), on=["RINPERSOON1"], suffixes=["","1"], validate="m:1")
print("New shape", bo.shape)
bo = pd.merge(bo, vo.rename(columns={"RINPERSOON":"RINPERSOON2"}), on=["RINPERSOON2"], suffixes=["","2"], validate="m:1")
print("Final shape", bo.shape)

print("Saving")
bo.to_csv(f"{data_path}/matching_analysis.tsv", sep="\t", index=None)


# Addresses of students in 2021 (make sure the students remain all year)
addressen = read_file_current_version("G:\Bevolking\GBAADRESOBJECTBUS", 2021)
display(addressen.head(1))
# drop rows before 2021
addressen = addressen.loc[addressen["GBADATUMAANVANGADRESHOUDING"]<"20210000"]


# Find our sample (students transitioning) and keep their last address
students = set(np.unique(np.concatenate([bo["RINPERSOON1"],bo["RINPERSOON2"]])))
addressen = addressen.loc[addressen["RINPERSOON"].isin(students),["RINPERSOON","RINOBJECTNUMMER"]]
addressen = addressen.drop_duplicates(subset=["RINPERSOON"], keep="last")

len(students), len(addressen)


# Merge addresses to coordinates (100x100 square)
coord = read_file_current_version("G:\BouwenWonen\VSLVIERKANTTAB", 2022)
coord = coord.loc[:, ["RINOBJECTNUMMER","VRLVIERKANT100M"]].drop_duplicates()
addressen = pd.merge(addressen, coord.loc[:, ["RINOBJECTNUMMER","VRLVIERKANT100M"]], validate = "m:1")
display(addressen.head(2))


print("\nStarting to merge sample to addresses, original shape")
print(bo.shape)
bo = pd.merge(bo, addressen.rename(columns={"RINPERSOON":"RINPERSOON1"}), on=["RINPERSOON1"], suffixes=["","1"], validate="m:1")
print("New shape", bo.shape)
bo = pd.merge(bo, addressen.rename(columns={"RINPERSOON":"RINPERSOON2"}), on=["RINPERSOON2"], suffixes=["","2"], validate="m:1")
print("Final shape", bo.shape)


# Calculate all distances of those transitioning (function in common_functions)
bo = calculate_distance(bo)

# Check the distribution of idstances
sns.displot(np.log10(1+bo["distance"].values))


## GROUP 1: Baseline

# Create baseline (we sort them by gemeente to increase the probability that the students live nearby)
baseline = pd.concat([
                    bo[['ONDERWIJSNR_crypt1','RINPERSOONS1', 'RINPERSOON1',
                        'RINObjectnummer','gemcode', 'POSTCODE', 'PLAATSNAAM', 'RINPERSOONS', 'ONDERWIJSNR_crypt',
                        'BRIN_crypt1', 'OPLNR', 'AANVINSCHR', 'EINDINSCHR', 'TYPEONDERWIJS',
                        'BRINVEST1', 'VOLEERJAAR', 'diff', 'year', 'month', 'RINObjectnummer1',
                        'gemcode1', 'POSTCODE1', 'PLAATSNAAM1', 'RINOBJECTNUMMER',"VRLVIERKANT100M"]].sort_values(by=["gemcode1"], ignore_index=True),
                    
                    bo[['ONDERWIJSNR_crypt2', 'RINPERSOONS2', 'RINPERSOON2','BRIN_crypt2', 'OPLNR2', 'AANVINSCHR2',
                        'EINDINSCHR2', 'TYPEONDERWIJS2', 'BRINVEST2', 'VOLEERJAAR2', 'diff2',
                        'year2', 'month2', 'RINObjectnummer2', 'gemcode2', 'POSTCODE2',
                        'PLAATSNAAM2',  'RINOBJECTNUMMER2',"VRLVIERKANT100M2"]].sample(frac=1, random_state=0).sort_values(by=["gemcode2"], ignore_index=True)
                    ],
                    axis = 1)

#Make sure we don't have students going to VO together
baseline = baseline.loc[baseline["BRIN_crypt1"] != baseline["BRIN_crypt2"]]

#Make sure we don't have students going to BO together (merge with original data)
baseline = pd.merge(baseline, bo[["RINPERSOON1","RINPERSOON2"]], how="left", indicator=True)

print(baseline["_merge"].value_counts())

baseline = baseline.loc[baseline["_merge"]=="left_only"]

#Add distance
baseline = calculate_distance(baseline)


## Checks to make sure it worked
print((baseline["gemcode1"] == baseline["gemcode2"]).value_counts())
print((bo["gemcode1"] == bo["gemcode2"]).value_counts())


print((baseline["POSTCODE1"] == baseline["POSTCODE2"]).value_counts())
print((bo["POSTCODE1"] == bo["POSTCODE2"]).value_counts())

# Save to temp
pickle.dump(bo, open(f"{temp_data}/student_pairs.csv", "wb+"))
pickle.dump(baseline, open(f"{temp_data}/student_pairs_baseline.csv", "wb+"))

bo.to_csv(f"{temp_data}/student_pairs.csv", sep="\t", index=None) 
baseline.to_csv(f"{temp_data}/student_pairs_baseline.csv", sep="\t", index=None) 





