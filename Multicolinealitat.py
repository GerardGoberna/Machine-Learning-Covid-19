import missingno as msn
from statsmodels.stats.outliers_influence import variance_inflation_factor


X = dataset.iloc [:, 0:18]
y = dataset.iloc [:, 18]

variance_df =pd.DataFrame()
variance_df["Vif"]=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
variance_df["Variables"]=X.columns

print(variance_df)