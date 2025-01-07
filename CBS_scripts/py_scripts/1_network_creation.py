#!/usr/bin/env python
# coding: utf-8

#Imports
import os
import numpy as np
import pandas as pd

# Functions to read and project the network
from common_functions import *

pd.options.display.max_columns = 100
pd.options.mode.chained_assignment = None

# # School network files
# Files:
# - GBAADRESOBJECTBUS (2019--) --> People's address (RINSOBJECTNUMMER)
# - Brinaddressen (2009--) --> Address of BRINs
# - Onderwijsdeelnemerstab (2000--) --> Student participations --> Basis and special
# - Onderwijsinschrtab (2000--) --> Student registrations --> Voortgezet and uni
# 
# For both education files:
# - RINPERSOON(S) --> ID
# - ONDERWIJSNR_crypt --> ID
# 
# Education levels:
# - (S)BO: (speciaal) basisonderwijs
# - (V)SO: (voortgezelt) speciaal oderwijs
# - VO: Voortgezet onderwijs
# - MBO middelbaar beroepsonderwijs
# - HO: hoger onderwijs
# 
# Algorithm:
# ```
# for every year:
#   for each type of network (basis, special, etc):
#     filter data
#     create network
# 
# - G:\Bevolking\GBAADRESOBJECTBUS
# - G:\Onderwijs\ONDERWIJSDEELNEMERSTAB
# - G:\Onderwijs\ONDERWIJSINSCHRTAB
# - G:\Onderwijs\BRINADRESSEN
 
bipartite_data_path = "H:/data_overload/network_creation/data/bipartite" 
projected_data_path = "H:/data_overload/network_creation/data/projected"

## Read data into memory
# Path definitions and global variables
#path_education_registration = "G:/Onderwijs/ONDERWIJSINSCHRTAB"
vars_education_vo = ["RINPERSOONS", #if RINPERSOON available or not
              "RINPERSOON",  #ID to link to other data
              "ONDERWIJSNR_crypt",  #student id
              "BRIN_crypt", #education site id
              "OPLNR", #type of education
              "AANVINSCHR", #start registration
              "EINDINSCHR", #end registration (99999999 = None)
              "TYPEONDERWIJS", #type education (BO, VO, etc)
              "VOBRINVEST", #branh of the education site
             "VOLEERJAAR", #year of educaiton
             ]

vars_education_bo = ["RINPERSOONS", #if RINPERSOON available or not
              "RINPERSOON",  #ID to link to other data
              "ONDERWIJSNR_crypt",  #student id
              "WPOBRIN_crypt", #education site id
              "WPOOPLNR", #type of education
              #"AANVINSCHR", #start registration
              #"EINDINSCHR", #end registration (99999999 = None)
              "WPOTYPEPO", #type education (BO, SBO, SO, VSP, NVTc)
              "WPOBRINVEST", #branh of the education site
             "WPOLEERJAAR", #year of educaiton
                     "WPOVERBLIJFSJRBO", # number of years in the school
                     "WPODENOMINATIE" #type of school, algemeen, antroposofish, et
             ]
    

#basis = ['B 00101 basisonderwijs groep 3-8', 'B 00100 basisonderwijs groep 1-2', 'B 00200 speciaal basisonderwijs groep 1-2', 'B 00201 speciaal basisonderwijs groep 3-8']
for year in range(2000,2022):
    education_registration = read_file_current_version("G:\Onderwijs\ONDERWIJSINSCHRTAB", year)
    education_registration["AANVINSCHR"] = education_registration["AANVINSCHR"].astype(int)
    education_registration["EINDINSCHR"] = education_registration["EINDINSCHR"].replace("Niet uitgeschreven",99999999).astype(int) #Consistency between years
    if "VO" in education_registration["TYPEONDERWIJS"].cat.categories:
        #Voortzet
        basis = filter_education(education_registration, vars_education_vo, type_ed = "VO")
        print(f"Secondary size {len(basis)}")
        basis = basis.sort_values(by=["BRIN_crypt","VOBRINVEST","VOLEERJAAR","OPLNR","ONDERWIJSNR_crypt","RINPERSOONS","RINPERSOON"])
        basis.to_csv(f"{bipartite_data_path}/vo_{year}.tsv", sep="\t", index=None)
    
    # BO
    education_registration = read_file_current_version("G:\Onderwijs\INSCHRWPOTAB", year)
    if education_registration is not None:
        #Some years are not as categories and have a different naming, make sure it's consistent
        education_registration["WPOTYPEPO"] =  pd.Categorical(education_registration["WPOTYPEPO"])
        education_registration["WPOTYPEPO"] = education_registration["WPOTYPEPO"].cat.rename_categories({'Basisonderwijs':"BO", 'Speciaal Basisonderwijs':"SBO"})
    
        #Voortzet
        basis = education_registration.loc[education_registration["WPOTYPEPO"] == 'BO', vars_education_bo]

        print(f"Primary size {len(basis)}")
        basis = basis.sort_values(by=["WPOBRIN_crypt","WPOBRINVEST","WPOLEERJAAR","WPOOPLNR","ONDERWIJSNR_crypt","RINPERSOONS","RINPERSOON"])
        basis.to_csv(f"{bipartite_data_path}/bo_{year}.tsv", sep="\t", index=None)
    
        


## Save data for students in 8th grade (last year of primary school)
#We will need this file when we are creating student pairs
df = pd.read_csv(f"{bipartite_data_path}/bo_2019.tsv", sep="\t", dtype=str)
df = df.loc[(df["WPOLEERJAAR"]==" 8")&(df["WPOVERBLIJFSJRBO"]==" 8")]
df.to_csv(f"{bipartite_data_path}/bo_2019_last_year.tsv", sep="\t", index=None)


## Project the unipartite networks (student-school) to bipartite (student-student)
files_to_proyect = [_ for _ in os.listdir(bipartite_data_path) if (".tsv" in _)][::-1]
files_to_proyect = [_ for _ in files_to_proyect if "2020" in _] #Only 2020 is needed for this paper


for file in files_to_proyect:
    if "bo" in file:
        columns_school = ["WPOBRIN_crypt","WPOBRINVEST","WPOLEERJAAR","WPOOPLNR","WPODENOMINATIE"]
        year_var = "WPOLEERJAAR"
    else:
        columns_school = ["BRIN_crypt","VOBRINVEST","VOLEERJAAR","OPLNR"]
        year_var = "VOLEERJAAR"
    path = f"F:/data_overload/network_creation/data/bipartite/{file}"
    project_network(path, f"{projected_data_path}/{file}", columns_school, year_var)






