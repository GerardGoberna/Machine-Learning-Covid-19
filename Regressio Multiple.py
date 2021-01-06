import statsmodels.formula.api as smf
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# Creem un
reg = linear_model.LinearRegression()

X = dataset.iloc [:, 0:18]
y = dataset.iloc [:, 18]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.33)

#Fem el model
X_train = sm.add_constant(X_train, prepend=True)
model_original = sm.OLS(endog=y_train, exog=X_train,)
model_original = model_original.fit()
print(model_original.summary())
print(model_original.params)


#Farem una regressió només amb les variables que tenen una correlació més alta
X2 = dataset.iloc [:, [0,1,2,3,13,14,15]]
y2 = dataset.iloc [:, 18]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, random_state=0, test_size=0.33)

X2_train = sm.add_constant(X2_train, prepend=True)
model_reduit = sm.OLS(endog=y2_train, exog=X2_train,)
model_reduit = model_reduit.fit()
print(model_reduit.summary())

#fem una comparació entre el model amb totes les variables i el model reduit
from statsmodels.stats.anova import anova_lm

anova_lm(model_original, model_reduit)

#Farem una regressió sense les variables que tenen un coeficient parcial de regressió alt
X2 = dataset.iloc [:, [0,1,2,3,4,5,7,8,9,10,11,13,14,15,16]]
y2 = dataset.iloc [:, 18]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, random_state=0, test_size=0.33)


X2_train = sm.add_constant(X2_train, prepend=True)
model_reduit = sm.OLS(endog=y2_train, exog=X2_train,)
model_reduit = model_reduit.fit()
print(model_reduit.summary())

#fem una comparació entre el model amb totes les variables i el model reduit
from statsmodels.stats.anova import anova_lm

anova_lm(model_original, model_reduit)