from sklearn.metrics import mean_squared_error
from mssql_session_util import sql_connection
from common_methods import *

import pickle
import warnings
warnings.filterwarnings('ignore')

nobs = 100

cnxn = sql_connection

# Reading the data from SQL
query = "SELECT LeadID, DateAdded, LeadSourceGroup, LoanPurpose, ZipCode FROM VLF.DimLead WHERE isActive = 1"
leads = pd.read_sql(query, cnxn, parse_dates='DateAdded')
print('Shape of Velocify leads data set:', leads.shape)

# Processed data frame
data = compile_df(leads)
print('Shape of Processed leads data set:', data.shape)

# Re-sampling the data to day level
data_daily = data.resample('1D').pad()
print('Shape of Re-sampled data set:', data.shape)

# Sorting the Data Frame and excluding the outliers
temp_df = data_daily.sort_values(['Direct Mail', 'Other_ZipCodes', 'Refinance'], ascending = False).iloc[15:, :]

# Imputing the outlier values with previous day values
temp_df_daily = temp_df.resample('1D').pad()

# Dropping 'Other_Campaigns','Other_LoanPurposes','Other_ZipCodes' from the data set
leads_df_var = temp_df_daily.drop(['Other_Campaigns','Other_LoanPurposes','Other_ZipCodes'], axis =1)


df_train, df_test = leads_df_var[0:-nobs], leads_df_var[-nobs:]
print("Training data set size :", df_train.shape, "Testing data set size :", df_test.shape)

# Checking the Stationarity on Training Data
df_train_diff = df_train.diff().dropna()
test_statistics_diff = test_stats(df_train_diff)
# test_statistics_diff

# Fitting the Model (VAR) on training Data Set
model = model(df_train_diff)

forecast = forcast_df(model, leads_df_var, nobs)

df_results = invert_transformation(df_train, forecast, second_diff=False)
fcast = df_results.iloc[:, 13:]
fcast.columns = leads_df_var.columns
for item in leads_df_var.columns:
    print("RMSE Score for '{}' :".format(item), (np.sqrt(mean_squared_error(df_test[item], fcast[item]))))

# Refitting the Model for Complete Data
leads_df_var_diff = leads_df_var.diff().dropna()
test_statistics_diff = test_stats(leads_df_var_diff)
# test_statistics_diff

model_refit = model(leads_df_var_diff)

# Saving the Model to disk
filename = 'TimeSeriesModel.sav'
pickle.dump(model_refit, open(filename, 'wb'))





