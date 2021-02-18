# Import packages
import ibm_db
import ibm_db_dbi as dbi
import pandas as pd


# Once the data are collected in the database they can be retrieved using the following function
# This connection object is used to access your data and contains your credentials or project token.
# You might want to remove those credentials before you share your notebook.
def download_dataset():
    Db2_j5_dsn = 'DATABASE={};HOSTNAME={};PORT={};PROTOCOL=TCPIP;UID={uid};PWD={pwd};SECURITY=SSL'.format(
        'Key information non published on Github',
        'To get information on how to download data please take a look on the main.py file',
        'Otherwise please contact me',
        uid='',
        pwd='')
    Db2_j5_connection = dbi.connect(Db2_j5_dsn)
    query = 'SELECT * FROM "CJH93586"."SN_20172994_IOT"'
    data_df_1 = pd.read_sql_query(query, con=Db2_j5_connection)
    Db2_j5_connection.close()
    return data_df_1
