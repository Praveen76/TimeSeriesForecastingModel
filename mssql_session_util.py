# SQL Connection

import pyodbc
import warnings
warnings.filterwarnings('ignore')

server = 'aspire-devops-sqlinstance.public.0fb73495fad6.database.windows.net,3342'
database = 'Marketing_Analytical_Data_mart_QA'
username = 'devopsadmin'
password = 'UH3$9r8nr3DuugDcq4g'   
driver= '{ODBC Driver 17 for SQL Server}'

def sql_connection():
	connection  = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
	return connection


