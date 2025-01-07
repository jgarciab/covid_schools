import csv
import os
import numpy as np
import pandas as pd
import time

from itertools import combinations


def read_file_current_version(path, year, usecols=None, nrows=-1):
    """
    Reads the most current version of an SPSS file (SAV format) from a specified folder.

    This function searches through a directory for files that contain the specified year 
    in their filenames and attempts to load the first matching SPSS file. If the file is 
    in SAV format, it will be read using either `pandas.read_spss` or `pyreadstat` 
    (with Latin-1 encoding fallback). If no SAV file is found, or if the file format is 
    unsupported, an exception is raised.

    Args:
        path (str): The path to the directory containing the files.
        year (int): The year to search for in the filenames.
        usecols (list): List of columns to keep. default=None (all)
        nrows (int): Number of rows to read. default=-1 (all)
    Returns:
        pandas.DataFrame: A DataFrame containing the data from the SPSS file.

    Raises:
        Exception: If a matching file is not in SAV format or if an unsupported file format is encountered.

    Notes:
        - Ensure that `pyreadstat` is installed as a fallback for reading SAV files with non-default encodings.
        - The function will stop after finding the first matching file containing the specified year.
    """
    # Loop through all files in the specified directory
    for file in os.listdir(path):
        # Check if the filename contains the specified year
        if str(year) in file:
            print(file)  # Debugging/confirmation log of the matching file
            
            # Check if the file extension suggests it is an SPSS SAV file
            if "SAV" in file.upper():
                try:
                    # Attempt to read the SAV file using pandas
                    return pd.read_spss(f"{path}/{file}")
                except Exception as e:
                    # If pandas fails, fall back to pyreadstat with explicit encoding
                    import pyreadstat
                    df, meta = pyreadstat.read_sav(f"{path}/{file}", encoding="latin1", usecols=usecols, row_limit=nrows)
                    return df
            else:
                # Raise an exception if the file format is unsupported
                raise Exception("File not in SPSS format. Code for other formats is not implemented.")


def filter_education(education_registration, vars_education, type_ed="BO basisonderwijs"):
    """
    Filters education registration data for a specific type of education and calculates additional columns.

    This function filters a dataset to include only registrations for a specific education type, 
    calculates the difference in registration dates, adjusts the academic year based on registration month, 
    and filters out records where the registration period is less than six months.

    Args:
        education_registration (pd.DataFrame): The dataset containing education registration details.
        vars_education (list): List of column names to retain in the filtered dataset.
        type_ed (str, optional): The type of education to filter by. Defaults to "BO basisonderwijs".

    Returns:
        pd.DataFrame: A filtered DataFrame containing the relevant education registration data, 
                      with additional columns for date differences and adjusted academic year.

    Notes:
        - Assumes `AANVINSCHR` (start date) and `EINDINSCHR` (end date) are in YYYYMMDD format.
        - The difference between dates is calculated in days.
        - Records with registration durations less than six months are removed.
        - The academic year is adjusted for registrations occurring before August.

    Raises:
        KeyError: If required columns (`AANVINSCHR`, `EINDINSCHR`, etc.) are missing from the input DataFrame.

    Example:
        >>> filtered_data = filter_education(df, ["AANVINSCHR", "EINDINSCHR"], "BO basisonderwijs")
    """
    # Filter dataset for the specified type of education and selected columns
    basis = education_registration.loc[
        education_registration["TYPEONDERWIJS"] == type_ed, vars_education
    ]
    
    # Calculate the duration of registration in days
    basis["diff"] = basis["EINDINSCHR"] - basis["AANVINSCHR"]

    # Extract the year and month from the start date (AANVINSCHR)
    basis["year"] = np.round(basis["AANVINSCHR"] / 10000)
    basis["month"] = np.round((basis["AANVINSCHR"] - basis["year"] * 10000) / 100)

    # Adjust the academic year for registrations occurring before August
    basis.loc[basis["month"] < 8, "year"] -= 1

    # Filter out records where the registration duration is less than six months
    num_removed = np.sum(basis["diff"] < 6000)
    print(f"Filtered {num_removed} observations that were not registered for at least 6 months.")
    basis = basis.loc[basis["diff"] > 6000]

    return basis



def project_network(path, path_save_to, columns_school, year_var="year"):
    """
    Projects a network of connections between students based on shared school attributes.

    This function reads a tab-separated CSV file, groups data by year and school attributes, 
    and generates all possible pairs of students within the same school group. The output 
    includes these pairs and relevant school attributes, saved to a specified file.

    Args:
        path (str): Path to the input CSV file containing student and school data.
        path_save_to (str): Path to save the output file containing the projected network.
        columns_school (list): List of columns that define the grouping for school attributes.
        year_var (str, optional): The column name representing the year. Defaults to "year".

    Returns:
        None: The function writes the output directly to the specified file.

    Raises:
        KeyError: If required columns are missing from the input dataset.
        FileNotFoundError: If the input file path is invalid.

    Notes:
        - Input file should be tab-separated and must include student IDs (`ONDERWIJSNR_crypt`, 
          `RINPERSOONS`, `RINPERSOON`) and the year variable.
        - The output is saved in tab-separated format, with one row per student pair.

    Example:
        >>> project_network("input.csv", "output.csv", ["School_Name", "School_ID"])
    """
    print(f"Analyzing file {path}")

    # Read the input data
    data_full = pd.read_csv(path, sep="\t", keep_default_na=False)

    # Define the columns to include in the output
    columns = columns_school + [
        "ONDERWIJSNR_crypt1",
        "RINPERSOONS1",
        "RINPERSOON1",
        "ONDERWIJSNR_crypt2",
        "RINPERSOONS2",
        "RINPERSOON2"
    ]

    # Open the output file for writing
    with open(path_save_to, "w+") as fout:
        fout.write("\t".join(columns) + "\n")  # Write header

        # Group the data by the year variable
        for year, data in data_full.groupby(year_var):
            # Skip invalid or placeholder years
            if ("n.v.t." in str(year)) or (str(year).strip() == "0"):
                continue

            print(year)

            # Create all pairs of students within each school group
            data = data.groupby(columns_school).apply(
                lambda x: combinations(
                    x[["ONDERWIJSNR_crypt", "RINPERSOONS", "RINPERSOON"]].values, 2
                )
            )
            print(f"Pairs for {len(data)} schools")

            st = time.time()

            # Unstack the pairs into rows
            data = data.apply(pd.Series).stack().reset_index(level=-1, drop=True).reset_index()
            print(f"{data.shape[0]} pairs unstacked in {time.time() - st: 2.0f} seconds")

            # Concatenate IDs of both students into a single string separated by tabs
            data[0] = data[0].apply(lambda x: "\t".join(np.concatenate(x)))

            # Write the data to the output file without using quotes for better performance
            data.to_csv(
                fout, sep="\t", index=None, header=False, quoting=csv.QUOTE_NONE, escapechar=" "
            )

def read_rivm(only_positives=True):
    """
    Reads and processes RIVM test data, returning a dictionary of days since a reference date for each person.

    This function loads RIVM test data from an SPSS file, processes it to exclude records with missing 
    personal identifiers, converts date fields, and optionally filters for positive test results. 
    The output is a dictionary mapping personal identifiers to the number of days since January 1, 2020.

    Args:
        only_positives (bool, optional): If True, only includes positive test results. Defaults to True.

    Returns:
        dict: A dictionary where keys are personal identifiers (`RINPERSOON`) and values are days 
              from January 1, 2020.

    Notes:
        - Requires `DatumMonsterafname` for date calculations and `Testuitslag` for filtering test results.
        - Assumes that `RINPERSOON` is a unique identifier for individuals.

    Example:
        >>> rivm_dict = read_rivm(only_positives=True)
    """
    path_rivm = "G:\\Maatwerk\\CORONIT\\CoronIT_GGD_testdata_20210921.sav"

    # Load RIVM data from the specified SPSS file
    rivm = pd.read_spss(path_rivm)

    # Exclude records with missing or empty BSN identifiers
    rivm = rivm.loc[rivm["RINPERSOON"] != '""']

    # Convert test date to datetime format
    rivm["date"] = pd.to_datetime(rivm["DatumMonsterafname"])

    # Print the range of test dates for verification
    print(rivm["DatumMonsterafname"].min(), rivm["DatumMonsterafname"].max())

    # Convert "Testuitslag" to a binary column (True for positive results)
    rivm["Testuitslag"] = rivm["Testuitslag"] != "NEGATIEF"

    # Filter to include only positive test results if `only_positives` is True
    if only_positives:
        rivm = rivm.loc[rivm["Testuitslag"]]

    # Print the maximum number of tests per person for statistics
    stats = rivm.groupby("RINPERSOON")["Testuitslag"].count()
    print(stats.max())

    # Calculate the number of days from the reference date (January 1, 2020)
    rivm["days_from_start"] = pd.to_datetime("2020-01-01")
    rivm["days_from_start"] = (rivm["date"] - rivm["days_from_start"]).dt.days

    # Keep only unique combinations of person ID and days from start
    rivm = rivm[["RINPERSOON", "days_from_start"]].drop_duplicates()

    # Return a dictionary mapping person ID to days from start
    return rivm.set_index("RINPERSOON")["days_from_start"].to_dict()


def calculate_distance(bo):
    """
    Calculates the distance between two locations based on coordinate information.

    This function processes a DataFrame containing coordinate information for two locations 
    and calculates the Euclidean distance between them. The coordinates are extracted from 
    string fields, converted to meters, and adjusted for resolution.

    Args:
        bo (pd.DataFrame): A DataFrame containing columns `VRLVIERKANT100M` and `VRLVIERKANT100M2`, 
                           which represent location identifiers as strings.

    Returns:
        pd.DataFrame: The input DataFrame with additional columns:
            - `east1`, `north1`: Converted coordinates for the first location (in meters).
            - `east2`, `north2`: Converted coordinates for the second location (in meters).
            - `distance`: The calculated distance between the two locations, with an adjustment factor of 52.

    Notes:
        - The `VRLVIERKANT100M` and `VRLVIERKANT100M2` columns are expected to follow a specific 
          format where the first set of digits represents the east coordinate, and the second 
          set represents the north coordinate.
        - An adjustment factor of 52 is added to account for the average distance between points 
          in a 100x100m grid.

    """
    # Extract and convert east and north coordinates for the first location
    bo["east1"] = bo["VRLVIERKANT100M"].str[1:5].astype(float) * 100  # Convert to meters
    bo["north1"] = bo["VRLVIERKANT100M"].str[6:].astype(float) * 100

    # Extract and convert east and north coordinates for the second location
    bo["east2"] = bo["VRLVIERKANT100M2"].str[1:5].astype(float) * 100
    bo["north2"] = bo["VRLVIERKANT100M2"].str[6:].astype(float) * 100

    # Calculate the distance using the Euclidean formula and add 52 for resolution adjustment
    bo["distance"] = 52 + np.sqrt((bo["east1"] - bo["east2"])**2 + (bo["north1"] - bo["north2"])**2)

    return bo
