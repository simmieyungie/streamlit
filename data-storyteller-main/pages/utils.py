import numpy as np 
import pandas as pd 
from pandas.api.types import is_numeric_dtype
# from pandas_profiling import ProfileReport

def isCategorical(col):
    unis = np.unique(col)
    if len(unis)<0.1*len(col):
        return True
    return False

# def getProfile(data):
#     report = ProfileReport(data)
#     report.to_file(output_file = 'data/output.html')

def getColumnTypes(cols):
    Categorical=[]
    Numerical = []
    Object = []
    for i in range(len(cols)):
        if cols["type"][i]=='categorical':
            Categorical.append(cols['column_name'][i])
        elif cols["type"][i]=='numerical':
            Numerical.append(cols['column_name'][i])
        else:
            Object.append(cols['column_name'][i])
    return Categorical, Numerical, Object

def isNumerical(col):
    return is_numeric_dtype(col)

#my custom generator
def get_types(dataframe):

    #empty dictionary
    df_types = dataframe.dtypes
    
    #Get columns
    df_cols = dataframe.columns

    #empty dict
    cols_types = {}

    for cols, types in zip(df_cols, df_types):
            if types in ["int64", "float64"]:
                cols_types[cols] = "numerical"
            else:
                 cols_types[cols] = "categorical"
    
    result = pd.DataFrame(cols_types, index = [0]).transpose().reset_index().rename(columns = {"index" : "column_name", 0  : "type"})
    return result

def genMetaData(df):
    col = df.columns
    ColumnType = [] 
    Categorical = []
    Object = []
    Numerical = []
    for i in range(len(col)):
        if isCategorical(df[col[i]]):
            ColumnType.append((col[i],"categorical"))
            Categorical.append(col[i])
        
        elif is_numeric_dtype(df[col[i]]):
            ColumnType.append((col[i],"numerical"))
            Numerical.append(col[i])
        
        else:
            ColumnType.append((col[i],"object"))
            Object.append(col[i])

    return ColumnType

def makeMapDict(col): 
    uniqueVals = list(np.unique(col))
    uniqueVals.sort()
    dict_ = {uniqueVals[i]: i for i in range(len(uniqueVals))}
    return dict_

def mapunique(df, colName):
    dict_ = makeMapDict(df[colName])
    cat = np.unique(df[colName])
    df[colName] =  df[colName].map(dict_)
    return cat 

    
## For redundant columns
def getRedundentColumns(corr, y: str, threshold =0.1): 
    cols = corr.columns
    redunt = []
    k = 0
    for ind, c in enumerate(corr[y]):
        if c<1-threshold: 
            redunt.append(cols[ind])
    return redunt

def newDF(df, columns2Drop):
    newDF = df.drop(columns2Drop, axis = 'columns')
    return newDF

if __name__ == '__main__':
    df = {"Name": ["salil", "saxena", "for", "int"]}
    df = pd.DataFrame(df)
    print("Mapping dict: ", makeMapDict(df["Name"]))
    print("original df: ")
    print(df.head())
    pp = mapunique(df, "Name")
    print("New df: ")
    print(pp.head())