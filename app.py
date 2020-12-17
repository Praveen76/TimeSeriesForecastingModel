import numpy as np
import pandas as pd
import datetime

import matplotlib.pyplot as plt
import time
import math
import pickle

from mssql_session_util import sql_connection 
from common_methods import *

from flask import Flask, request, jsonify, render_template, url_for, render_template_string
import pickle


cnxn = sql_connection()

# Reading the data from SQL
query = "SELECT LeadID, DateAdded, LeadSourceGroup, LoanPurpose, ZipCode FROM VLF.DimLead WHERE isActive = 1"



def campaignForecast(tableName,df,cnxn):
    print('Size of Data :',df.shape)
    df.to_sql(tableName,cnxn)
	


def LoanPurposeForecast(tableName,df,cnxn):
    print('Size of Data :',df.shape)
    df.to_sql(tableName,cnxn)

def ZipCodeForecast(tableName,df,cnxn):
    print('Size of Data :',df.shape)
    df.to_sql(tableName,cnxn)
	


def storeResults(filterName,tableName,df,cnxn):
        switcher={
                'Campaign':campaignForecast,
                'LoanPurpose':LoanPurposeForecast,
                'ZipCode': ZipCodeForecast
                }
        func=switcher.get(filterName,lambda :'Invalid Entry')
        return func(tableName,df,cnxn)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('Index.html')

# def validate_prediction_input(input_value):
#     try:
#         required_value = int(input_value)
#         return "Success"
#     except Exception as e:
#         print("the exception that happened for the passed {} is {}".format(input_value, e))
#         return "Failure"

@app.route('/predict', methods = ['POST'])
def predict2():
    output_dict = {"error": None}
    obtained_values = [x for x in request.form.values()]

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("the received values are ", obtained_values)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    required_df = get_prediction(30, obtained_values[0])
    output_dict["data"] = required_df.to_html(index=False)

    return output_dict


    # if len(obtained_values) > 0:
    #     required_input = obtained_values[0]
    #     if validate_prediction_input(required_input) == "Failure":
    #         output_dict["error"] = "the passed input value is not an integer".format(required_input)
    #         return output_dict
    #
    #     # Success logic
    #     required_df = get_prediction(obtained_values)
    #     output_dict["data"] = required_df.to_html(index = False)
    #     return output_dict
    # else:
    #     output_dict["error"] = "No value posted to the API"
    #     return output_dict

#@app.route('/predict',methods=['POST'])
def get_prediction(int_features, obtained_values): # 30 days forecast

    if obtained_values == 'Campaign':
        filterName=obtained_values
        tableName=campaignForecast
        obtained_values = ['Internet', 'Radio', 'Social Media', 'TV']
    elif obtained_values == 'LoanPurpose':
        filterName=obtained_values
        tableName=LoanPurposeForecast
        obtained_values = ['Purchase', 'Refinance', 'Home Equity']
    elif obtained_values == 'ZipCode':
        filterName=obtained_values
        tableName=ZipCodeForecast
        obtained_values = ['75', '76', '77', '78', '79']

    target = obtained_values


    print("Entered Forecast days are {}".format(int_features))

    leads = pd.read_csv("Velocify_Leads.csv", parse_dates=['DateAdded'])
    # Processed data frame
    data = compile_df(leads)
    # print('Shape of Processed leads data set:', data.shape)

    # Resampling to Daily Level
    data_daily = data.resample('1D').pad()

    # Dropping 'Other_Campaigns','Other_LoanPurposes','Other_ZipCodes' from the data set
    leads_df_var = data_daily.drop(['Direct Mail', 'Other_Campaigns', 'Other_LoanPurposes', 'Other_ZipCodes'], axis=1)
    leads_df_var_diff = leads_df_var.diff().dropna()
    # Load pre-trained model
    filename = 'TimeSeriesModel.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    lag_order_refit = loaded_model.k_ar
    forecast_input_refit = leads_df_var_diff[-lag_order_refit:]
    print("Shape of forecast_input_refit: ", forecast_input_refit.shape)

    fc_refit = loaded_model.forecast(y=forecast_input_refit.values, steps=int_features)

    print("Shape of the forecast data set: ", fc_refit.shape)

    start_date = leads_df_var.index.max() + pd.Timedelta(days = 1)
    end_date = leads_df_var.index.max() + pd.Timedelta(days = int_features)
    ix = pd.date_range(start=start_date, end=end_date, freq='D')
    df_forecast_refit = round(pd.DataFrame(fc_refit, index = ix, columns=leads_df_var.columns))

    df_results_refit = invert_transformation(leads_df_var, df_forecast_refit, second_diff=False)
    df_results_refit = df_results_refit.iloc[:, 12:]
    df_results_refit.columns = leads_df_var.columns
    df_results_refit = df_results_refit.astype(int)
    df_results_refit[df_results_refit < 0 ] = 0
    df_results_refit = df_results_refit[target]
    df_results_refit.insert(0, 'Date', df_results_refit.index)
    df_results_refit['Date'] = pd.to_datetime(df_results_refit['Date']).dt.strftime('%Y-%d-%m')
    print('Shape of returned data :',df_results_refit.shape)
	
    #storeResults(filterName,tableName,df_results_refit,cnxn)

    # Plotting the Graph
    # temp_fcast = pd.concat([leads_df_var.tail(1), df_results_refit.drop('Date', axis=1)], axis=0)
    # temp_fcast = temp_fcast[target]
    # temp_fcast.columns = leads_df_var[target].columns + "_Forecast"
    #
    # temp_orgi = leads_df_var[target]
    #
    # fig, axes = plt.subplots(len(temp_orgi.columns), figsize=(20, 40), dpi=150)
    # for i, (col, ax) in enumerate(zip(temp_orgi.columns, axes.flatten())):
    #     temp_orgi[col][-6 * int_features:].plot(legend=True, ax=ax, title='Original Data')
    #     temp_fcast[str(col) + "_Forecast"].plot(legend=True, ax=ax, linestyle='dashed', color='red',
    #                                             title='Forecasted Data')
    #     ax.set_title(col + ": Forecast for next {} days from date {}".format(int_features,
    #                                                                          temp_orgi.index.max().strftime(
    #                                                                              "%Y-%d-%m")))
    #     ax.tick_params()
    #
    # plt.tight_layout();

    return df_results_refit

if __name__ == '__main__':
    app.run(debug=True)