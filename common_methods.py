# Common Methods

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR

# Grouping the Loan Purpose

def group_purpose(series):
    s = list()
    for v in series:
        v = str(v)
        if 'pur' in v.lower():
            s.append('Purchase')
        elif 'refi' in v.lower():
            s.append('Refinance')
        elif ('home' or 'equity') in v.lower():
            s.append('Home Equity')
        else:
            s.append('Other_LoanPurposes')
    return s

# Grouping the Campaign (Lead Source Group)

def group_campaign(series):
    s = list()
    campaign = ['TV', 'Radio', 'Internet', 'Direct Mail', 'Social Media']
    for v in series:
        if v not in campaign:
            s.append('Other_Campaigns')
        else:
            s.append(v)
    return s

# Grouping the Zip Code

def group_zip(series):
    s = list()
    zips = ['75', '76', '77', '78', '79']
    for v in series:
        if v not in zips:
            s.append('Other_ZipCodes')
        else:
            s.append(v)
    return s


# Preparing Data Set for Model

def compile_df(df):
    df.dropna(inplace = True)
    df['Date'] = df['DateAdded'].dt.date
    df['ZipCode'] = [str(zp)[:2] for zp in df['ZipCode']]
    df['ZipCode'] = group_zip(df['ZipCode'])
    df['LoanPurpose'] = group_purpose(df['LoanPurpose'])
    df['Campaign'] = group_campaign(df['LeadSourceGroup'])

    # Campaign
    Campaign = df.groupby(['Date','Campaign'])['DateAdded'].count().reset_index()
    Campaign = Campaign.pivot_table(index="Date",columns="Campaign", values="DateAdded", fill_value=0)

    # LoanPurpose
    LoanPurpose = df.groupby(['Date','LoanPurpose'])['DateAdded'].count().reset_index()
    LoanPurpose = LoanPurpose.pivot_table(index="Date",columns="LoanPurpose", values="DateAdded", fill_value=0)

    # ZipCode
    ZipCode = df.groupby(['Date','ZipCode'])['DateAdded'].count().reset_index()
    ZipCode = ZipCode.pivot_table(index="Date",columns="ZipCode", values="DateAdded", fill_value=0)

    # Joining to make Final DF
    df = Campaign.join(LoanPurpose,how="inner").join(ZipCode, how="inner")
    df.index = pd.DatetimeIndex(df.index)

    return df

# Checking for Stationarity of time Series

def test_stats(df):
    test_statistics = pd.DataFrame(columns=('method', 'adf stat', 'lags', 'p-val', '1%', '5%', '10%','stationary'),
                                   index=df.columns)
    test_statistics.loc[:,'stationary']=False

    for item in df.columns:
        result = adfuller(df.loc[:, item], autolag='AIC')
        test_statistics.loc[item]=pd.Series(data=['None',
                                                  result[0], result[2],
                                                  np.around(result[1],6),
                                                  result[4]['1%'], result[4]['5%'],
                                                  result[4]['10%'],result[0]<result[4]['1%']
                                                  ],
                                             name=item,
                                             index=test_statistics.columns)
    return test_statistics

# Inverse Transformation from first ordered differenced series

def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:
        # Roll back 2nd Diff
        #if second_diff:
        #    df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        # df_fc[col] = df_train[col].iloc[-1] + df_fc[col].cumsum()
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[col].cumsum()

    return df_fc

# Developing Model

def model(df):
    model = VAR(df)

    AIC = []
    for i in range(1,31,1):
        result = model.fit(i)
        AIC.append(result.aic)

    df_ = pd.DataFrame({'lag':range(1,31,1), 'aic':AIC})
    lag = df_[df_['aic'] == df_['aic'].min()]['lag'].values[0]
    print("The best lag value: ", lag)
    model_fitted = model.fit(lag)
    return model_fitted

def forcast_df(train_df, orgi_df, steps):
    df = model(train_df)
    lag_order = df.k_ar
    forecast_input = df.values[-lag_order:]
    fc = df.forecast(y=forecast_input, steps=steps)
    df_forecast = round(pd.DataFrame(fc, index=orgi_df.index[-steps:], columns=orgi_df.columns))
    print("Size of Forecasted Data Frame: ", df_forecast.shape)
    return df_forecast























