import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle


st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Churn Prediction")

st.header('Dataset Selection and Loading')
dataset_name = st.selectbox("Select Datasets", ['Telco_Churn'])
st.write('You selected:', dataset_name, 'dataset')


def get_dataset():
    df = pd.read_csv("telecom.csv")
    target_column = "Churn"
    df["SeniorCitizen"] = df["SeniorCitizen"].replace([0, 1], ['No', 'Yes'])
    df["TotalCharges"] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    return df, target_column


df, target_column = get_dataset()
df1 = df.copy()

st.subheader('Checking the Dataset')
st.table(df.head())
st.subheader('Types of Variables')
pd.DataFrame(df.dtypes, columns = ["Col"])

st.subheader('Exploratory Data Analysis')
visualization = st.selectbox('Select Visualization Type', ('Pairplot', 'Heatmap'))
if visualization == 'Pairplot':
    fig = sns.pairplot(df, hue=target_column)
    st.pyplot(fig=fig)
elif visualization == 'Heatmap':
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), vmin=-1, cmap='coolwarm', annot=True)
    st.pyplot(fig)

st.header('Columns Selection for Analysis')
st.cache()
columns_select = st.selectbox("Select All Columns or Select Columns You Want to Drop",
                              ("All Columns", "Select Columns to Drop"))
if columns_select == "Select Columns to Drop":
    d_list = st.multiselect("Please select columns to be dropped", df.columns)
    df = df.drop(d_list, axis=1)
    if st.button("Generate Codes for Columns to be Dropped"):
        if columns_select == "Select Columns to Drop":
            st.code(f"""df = df.drop({d_list},axis=1)""")

st.header('Multicollinearity Checking')
threshold = st.selectbox("Select threshold for multicollinearity", (1, 0.9, 0.8, 0.7))
Corr_Columns = []
numeric_cols = df.select_dtypes(exclude=["object"]).columns


def multicollinearity(dataset, threshold):
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                Corr_Columns.append(colname)
    return Corr_Columns


multicollinearity(df, threshold=threshold)

df.drop(Corr_Columns, axis=1, inplace=True)
if len(Corr_Columns) > 0:
    st.write(f"{Corr_Columns} columns having correlation value more than {threshold} and therefore dropped")
else:
    st.write("No columns found exceeding the threshold")

st.header('Checking Missing Values')
threshold_value = st.selectbox("Select Threshold for Missing Value Checking", (70, 60, 50, 40, 30, 20, 10))
drop_list = []


def missing_drop(df, threshold_value):
    for variable in df.columns:
        percentage = round((df[variable].isnull().sum() / df[variable].count()) * 10)
        if percentage > threshold_value:
            st.write(f"The percentage of missing variable for {variable} is % {percentage}")
            drop_list.append(variable)
    if len(drop_list) == 0:
        st.write("No Columns exceeding the Threshold")


missing_drop(df, threshold_value)

st.header('Missing Value Handling')
missing_values = st.selectbox("Select one method for missing value imputation ",
                              ("Drop All Missing Values", "Filling with Either Median or Mode"))


def impute_with_median(df, variable):
    if df[variable].isnull().sum() > 0:
        st.write(
            f"{variable} column has {df[variable].isnull().sum()} missing value and replaced with median:{df[variable].median()}")
        df[variable + "NAN"] = np.where(df[variable].isnull(), 1, 0)
        df[variable] = df[variable].fillna(df[variable].median())


def impute_with_mode(df, variable):
    if df[variable].isnull().sum() > 0:
        st.write(
            f"{variable} column has {df[variable].isnull().sum()} missing value and replaced {df[variable].mode()[0]}")
        df[variable + "NAN"] = np.where(df[variable].isnull(), 1, 0)
        frequent = df[variable].mode()[0]
        df[variable].fillna(frequent, inplace=True)


if missing_values == "Drop All Missing Values":
    for variable in df.columns:
        if df[variable].isnull().sum() > 0:
            st.write(f"{variable} column has {df[variable].isnull().sum()} missing value")
    st.write(f" percentages of total missing data :% {(df.isnull().sum().sum() / df.shape[0]) * 10}")
    df.dropna(inplace=True)

else:
    for i in df.columns:
        if np.dtype(df[i]) == "object":
            impute_with_mode(df, i)
        else:
            impute_with_median(df, i)

st.header('Outliers Detection and Handling')

Handling_Outliers = st.selectbox("Select One Option for Outlier Handling ", ("Keep Outliers", "Handle Outliers"))
numeric_cols = df.select_dtypes(exclude=["object"]).columns


def outliers_detection_handling(df, variable):
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
    lower_bound = df[variable].quantile(0.25) - (IQR * 1.5)
    upper_bound = df[variable].quantile(0.75) + (IQR * 1.5)
    df[variable] = np.where(df[variable] > upper_bound, upper_bound, df[variable])
    df[variable] = np.where(df[variable] < lower_bound, lower_bound, df[variable])
    return df[variable]


if Handling_Outliers == "Handle Outliers":
    for i in numeric_cols:
        IQR = df[i].quantile(0.75) - df[i].quantile(0.25)
        lower_bound = df[i].quantile(0.25) - (IQR * 1.5)
        upper_bound = df[i].quantile(0.75) + (IQR * 1.5)
        num_outliers = df[~df[i].between(lower_bound, upper_bound)].value_counts().sum()
        if (df[i].max() > upper_bound) | (df[i].min() < lower_bound):
            outliers_detection_handling(df, i)
            st.write(f"{i}  column has {num_outliers} outliers and set to either upper or lower bound.")
else:
    st.write("You are keeping all outliers")




st.header('One Hot Encoding')
y = df[target_column]
X = df.drop(target_column, axis=1)
df = X
encode_list = []


def one_hot_encoder(df):
    for i in X.columns:
        if (np.dtype(df[i]) == "object"):
            unique_value = len(df[i].value_counts().sort_values(ascending=False).head(10).index)
            if unique_value > 10:
                for categories in (df[i].value_counts().sort_values(ascending=False).head(10).index):
                    df[i + "_" + categories] = np.where(df[i] == categories, 1, 0)
                    encode_list.append(i + "_" + categories)

            else:
                for categories in (df[i].value_counts().sort_values(ascending=False).head(unique_value - 1).index):
                    df[i + "_" + categories] = np.where(df[i] == categories, 1, 0)
                    encode_list.append(i + "_" + categories)

    return df, encode_list


num_cat_col = len(df.select_dtypes(include=["object"]).columns)
one_hot_encoder(df)

for i in df.columns:
    if (np.dtype(df[i]) == "object"):
        df = df.drop([i], axis=1)
col_after_endoded_all = df.columns
st.write(f"One hot encoding : {num_cat_col} columns are encoded and  {len(encode_list)} new columns are added")




st.header('Feature Engineering')
feature_selection = st.selectbox("Feature Selection", ("Keep all Features", "Select Features"))


def feature_importance(endogenous, exogenous, n):
    selected_feat = SelectKBest(score_func=chi2, k=10)
    selected = selected_feat.fit(endogenous, exogenous)
    feature_score = pd.DataFrame(selected.scores_, index=endogenous.columns, columns=["Score"])["Score"].sort_values(
        ascending=False).reset_index()
    st.write("Table: Feature Importance")
    st.table(feature_score.head(n))
    st.write("Feature Importance")
    plt.figure(figsize=(20, 12))
    sns.barplot(data=feature_score.sort_values(by='Score', ascending=False).head(n), x='Score', y='index')
    st.pyplot()
    X_Selected.extend(feature_score["index"].head(n))


if feature_selection == "Select Features":
    st.write()
    feature_number = st.slider("Select Number of Features You Want to Observe by Using Slider", 1, df.shape[1])
    X_Selected = []
    feature_importance(df, y, feature_number)
    df = df[X_Selected]
else:
    st.write(f"You selected all features and the number of selected features is: {len(df.columns)}")

st.header('Standardization')
standardization = st.selectbox("Standardization", ("No Need to Apply Standardization", "Apply Standardization"))
if standardization == 'Apply Standardization':
    methods = st.selectbox("Select one of the Standardization Methods: ",
                           ("StandardScaler", "MinMaxScaler", "RobustScaler"))

    num_cols = []
    df_new = df1.drop(target_column, axis=1)
    numeric_cols = df_new.select_dtypes(exclude=["object"]).columns
    for i in numeric_cols:
        if i not in df.columns:
            pass
        else:
            num_cols.append(i)

    for i in num_cols:
        if methods == "StandardScaler":
            scaler = StandardScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])

        elif methods == "MinMaxScaler":
            scaler = MinMaxScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])

        else:
            scaler = RobustScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])
    st.write(f"{num_cols} are scaled by using {methods}")

st.header('Model Evaluation')
models_name = st.sidebar.selectbox("Select Model for ML", ("LightGBM", "Random Forest"))
params = dict()


def parameter_selection(clf_name):
    if clf_name == "LightGBM":
        st.sidebar.write("Select Parameters")
        n_estimators = st.sidebar.slider("n_estimators", 10, 1000, 100, 10)
        max_depth = st.sidebar.slider("max_depth", 3, 20,1,1)
        learning_rate = st.sidebar.slider('learning_rate', 0.01, 1.0, 0.1, 0.1)
        num_leaves = st.sidebar.slider('num_leaves', 10, 300, 31, 10)
        params["n_estimators"] = n_estimators
        params["max_depth"] = max_depth
        params["num_leaves"] = num_leaves
        params["learning_rate"] = learning_rate

    else:
        st.sidebar.write("Select Parameters")
        max_depth = st.sidebar.slider("max_depth", 2, 100, 1, 2)
        n_estimators = st.sidebar.slider("n_estimators", 50, 1000, 100, 10)
        criterion = st.sidebar.selectbox("criterion", ('gini','entropy'))
        max_features = st.sidebar.selectbox("max_features", ('auto', 'sqrt', 'log2'))
        min_samples_leaf = st.sidebar.slider("min_samples_leaf", 1, 10, 5)
        min_samples_split = st.sidebar.slider("min_samples_split", 2, 10)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        params["criterion"] = criterion
        params["max_features"] = max_features
        params["min_samples_leaf"] = min_samples_leaf
        params["min_samples_split"] = min_samples_split
        a = RandomForestClassifier()
    return params


parameter_selection(models_name)


def get_classifier(classifier_name, params):

    if classifier_name == "LightGBM":

        clf_model = LGBMClassifier(num_leaves=params["num_leaves"], n_estimators=params["n_estimators"], max_depth=params["max_depth"], learning_rate=params["learning_rate"], random_state=42)
    else:
        clf_model = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                                            criterion=params["criterion"], max_features=params["max_features"],
                                            min_samples_leaf=params["min_samples_leaf"],
                                            min_samples_split=params["min_samples_split"], random_state=42)

    return clf_model


clf_model = get_classifier(models_name, params)
st.cache()

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
clf_model.fit(X_train, y_train)
y_pred = clf_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1_scr = f1_score(y_test, y_pred, average='macro')
cls_report = classification_report(y_test, y_pred)
cnf_matrix = confusion_matrix(y_test, y_pred)
st.sidebar.write(f"Selected Classifier = {models_name}")
st.sidebar.write(f"Accuracy Score = {acc}")
st.write(f"Selected Classifier = {models_name}")
st.write(f"Accuracy Score = {acc}")
st.subheader('Confusion Matrix')
st.table(cnf_matrix)
st.subheader('Classification Report')
st.text('Classification Report:\n ' + cls_report)

st.header('Model Download')
st.cache()


def download_model(model):
    pickle.dump(clf_model, open("final_model", 'wb'))


download_model(clf_model)
st.write(f'Trained model is saved as final_model for later use')




st.header('Churn Probability of Customers')
if st.button("Prediction Probality of Top 10 Churned Customers"):
    col_val = ['0', '1']
    a = clf_model.predict_proba(X_test)
    b = pd.DataFrame(data=a, columns=col_val)
    sorted_val = b.sort_values(by='0', ascending=False)
    st.table(sorted_val.head(10))


if st.button("Prediction Probality of Top 20 Loyal Customers"):
    col_val = ['0', '1']
    a = clf_model.predict_proba(X_test)
    b = pd.DataFrame(data=a, columns=col_val)
    sorted_val = b.sort_values(by='0')
    st.table(sorted_val.head(20))

st.header('Prediction')
st.subheader("Please select your variables to predict")
st.write('Please **do not leave** Tenure, Montly Charges, and Total Charges columns **empty**')
df1.drop(target_column, axis=1, inplace=True)
df1.drop('customerID', axis=1, inplace=True)
columns = df1.columns
cat_columns = df1.select_dtypes(include=["object"]).columns
dftest = pd.DataFrame()


for i in df1.columns:
    if i not in cat_columns:
        try:
            dftest[i] = [int(st.text_input(i))]
        except:
            dftest[i] = np.nan

    else:
        try:
            dftest[i] = [st.selectbox(i, df1[i].value_counts().index.tolist())]
        except:
            dftest[i] = np.nan



def prediction(dftest):
    try:
        dftest.drop(corCol, axis=1, inplace=True)
    except:
        pass

    try:
        dftest.drop(d_list, axis=1, inplace=True)
    except:
        pass

    def impute_with_median(df, variable):
        if df[variable].isnull().sum() > 0:
            df[variable + "NAN"] = np.where(df[variable].isnull(), 1, 0)
            df[variable] = df[variable].fillna(dftest[variable].median())

    def impute_with_mode(df, variable):
        if df[variable].isnull().sum() > 0:
            df[variable + "NAN"] = np.where(df[variable].isnull(), 1, 0)
            frequent = dftest[variable].mode()[0]
            df[variable].fillna(frequent, inplace=True)

    for i in dftest.columns:
        if (np.dtype(dftest[i]) == "object"):
            impute_with_mode(dftest, i)
        else:
            impute_with_median(dftest, i)

    def outliers_detection_handling(df, variable):
        IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
        lower_bound = df[variable].quantile(0.25) - (IQR * 1.5)
        upper_bound = df[variable].quantile(0.75) + (IQR * 1.5)
        df[variable] = np.where(df[variable] > upper_bound, upper_bound, df[variable])
        df[variable] = np.where(df[variable] < lower_bound, lower_bound, df[variable])
        return df[variable]

    try:
        if Handling_Outliers == "Handle Outliers":
            for i in numeric_cols:
                outliers_detection_handling(dftest, i)
    except:
        pass

    def one_hot_encoder(df):
        for i in encode_list:
            try:
                for categories in df[i]:
                    df[i] = np.where(df[i] == categories, 1, 0)
            except:
                df[i] = np.where(False, 1, 0)

        return df

    one_hot_encoder(dftest)

    for i in col_after_endoded_all:
        if i not in dftest.columns:
            dftest[i] = np.where(False, 1, 0)

    dftest = dftest.loc[:, col_after_endoded_all]
    dftest = dftest.drop(dftest.select_dtypes("object").columns, axis=1)

    if feature_selection == "Select Features":
        dftest = dftest[X_Selected]

    if standardization == 'Apply Standardization':
        try:
            dftest[num_cols] = scaler.fit_transform(dftest[num_cols])
        except:
            pass

    result = clf_model.predict(dftest)
    st.text('Predicted Result based on Data given is :' + ' ' + result)


if st.button("Predict"):
    prediction(dftest)
