# Import packages
import ibm_db
import ibm_db_dbi as dbi
import pandas as pd


# Once the data are collected in the database they can be retrieved using the following function
# This connection object is used to access your data and contains your credentials or project token.
# You might want to remove those credentials before you share your notebook.
def download_dataset():
    """
    Downloads the dataset from the IBM DB2 SQL database.

    :return: the dataset as a pandas dataframe.
    """
    # Parameters for the connection to the database
    Db2_j5_dsn = "API key not publicly published. For the assignment, the keys are saved in the config.py within the " \
                 "folder submitted on moodle."
    # Establish the connection
    Db2_j5_connection = dbi.connect(Db2_j5_dsn)
    # Query to process
    query = 'SELECT * FROM "CJH17542"."SN_20172994_IOT"'
    # Get data from the database through the query created
    data_df_1 = pd.read_sql_query(query, con=Db2_j5_connection)
    # Close the connection
    Db2_j5_connection.close()
    return data_df_1
