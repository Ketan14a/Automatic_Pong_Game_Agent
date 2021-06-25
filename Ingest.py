from pathlib import Path
import pandas as pd


##############################################
# Implement the below method
# The method should be dataset-independent
##############################################
def read_dataset(path: Path) -> pd.DataFrame:
    """
    This method will be responsible to read the dataset.
    Please implement this method so that it returns a pandas dataframe from a given path.
    Notice that this path is of type Path, which is a helper type from python to best handle
    the paths styles of different operating systems.
    """
    # Converting the path object into string for flexible parsing
    path_to_string = str(path)

    # Determining the file type and reading it through appropriate Pandas method
    if '.csv' in path_to_string or '.data' in path_to_string:
        dataset_data = pd.read_csv(path)

    elif '.html' in path_to_string:
        dataset_data = pd.read_html(path)

    elif '.json' in path_to_string:
        dataset_data = pd.read_json(path)

    elif '.xlsx' in path_to_string or '.xls' in path_to_string:
        dataset_data = pd.read_excel(path)

    elif '.sql' in path_to_string:
        dataset_data = pd.read_sql(path)

    elif '.dta' in path_to_string:
        dataset_data = pd.read_stata(path)

    elif '.hdf' in path_to_string:
        dataset_data = pd.read_hdf(path)

    elif '.pickle' in path_to_string or '.pkl' in path_to_string:
        dataset_data = pd.read_pickle(path)

    elif '.spss' in path_to_string:
        dataset_data = pd.read_spss(path)

    elif '.sas' in path_to_string:
        dataset_data = pd.read_sas(path)

    elif '.gbq' in path_to_string:
        dataset_data = pd.read_gbq(path)

    elif '.fwf' in path_to_string:
        dataset_data = pd.read_fwf(path)

    else:
        dataset_data = None

    # Returning the result
    return dataset_data


if __name__ == "__main__":
    """
    In case you don't know, this if statement lets us only execute the following lines
    if and only if this file is the one being executed as the main script. Use this
    in the future to test your scripts before integrating them with other scripts.
    """
    dataset = read_dataset(Path('..', '..', 'iris.csv'))
    assert type(dataset) == pd.DataFrame
    print("ok")
