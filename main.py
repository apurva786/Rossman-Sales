import pandas as pd
import numpy as np
import math
import time
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle


class main():

	def __init__(self):
		print("reading train.csv,test.csv and store.csv and putting them in pandas dataframe")
		self.test=pd.read_csv("test.csv")
		test=self.test
		self.train=pd.read_csv("train.csv")
		train=self.train
		self.store=pd.read_csv("store.csv")
		store=self.store
		store=self.pre_processing_store(store)
		train=self.pre_processing_train(train)
		self.test_train_date_and_merge(train,store)
		self.test_train_date_and_merge(test,store)
		self.parametr_tuning_xgboost(train)
		self.train_xgboost_model()
		self.save_in_csv()
		self.make_pickel()

	def models_with_default_parameters(self):
		print("predicting different models with their default parameters")
		Linear_regression()
		Decision_tree_regressor()
		Random_forest_regressor()


	def pre_processing_store(self,store):
		print("Pre-processing store file")
		print("Fill all Null values with -1")
		store.fillna(-1,inplace=True)
		print("label encoding StoreType")
		store.loc[store['StoreType']=='a', 'StoreType']='1'
		store.loc[store['StoreType']=='b', 'StoreType']='2'
		store.loc[store['StoreType']=='c', 'StoreType']='3'
		store.loc[store['StoreType']=='d', 'StoreType']='4'
		store['StoreType'] = store['StoreType'].astype(float)
		print("label encoding Assortment")
		store.loc[store['Assortment']=='a', 'Assortment']='1'
		store.loc[store['Assortment']=='b', 'Assortment']='2'
		store.loc[store['Assortment']=='c', 'Assortment']='3'
		store['Assortment'] = store['Assortment'].astype(float)
		print("Droping promotinterval")
		store.drop('PromoInterval',axis=1,inplace=True)
		return store


	def pre_processing_train(self,train):
		print("Pre-processing train")
		print("Considering only the stores which are opne and sales greater than 0")
		train = train[(train['Open']==1)&(train['Sales']>0)]
		print("Label encoding stateHoliday")
		train.loc[train['StateHoliday']=='a', 'StateHoliday']=1
		train.loc[train['StateHoliday']=='b', 'StateHoliday']=1
		train.loc[train['StateHoliday']=='c', 'StateHoliday']=1
		train.loc[train['StateHoliday']=='0', 'StateHoliday']=0
		print("We need not do the same for test as it contains only 1 value 0")
		return train


	def test_train_date_and_merge(self,process,store):
		print("Seperating day, month and year from date")
		for ds in [process]:
		    tmpDate = [time.strptime(x, '%Y-%m-%d') for x in ds.Date]
		    ds[  'mday'] = [x.tm_mday for x in tmpDate]
		    ds[  'mon'] = [x.tm_mon for x in tmpDate]
		    ds[  'year'] = [x.tm_year for x in tmpDate]
		process.drop('Date',axis=1,inplace=True)
		print("performing left outer join wiht store")
		process = process.merge(store, on = 'Store', how = 'left')

	def rmspe(y, yhat):
	    return np.sqrt(np.mean(((y - yhat)/y) ** 2))

	def rmspe_xg(yhat, y):
	    y = np.expm1(y.get_label())
	    yhat = np.expm1(yhat)
	    return "root_mean_square_error", rmspe(y, yhat)


	def parametr_tuning_xgboost(self,train):
		print("parameter tuning with xgboost")
		features = list(train.columns)
		features.remove('Sales')
		model = XGBRegressor()
		param_grid = {
		'max_depth':range(8,16,2),
		'min_child_weight':range(7,15,2),
		'learning_rate':[x/10 for x in range(1,4)],
		'subsample':[x/10 for x in range(5,9)],
		'colsample_bytree':[x/10 for x in range(5,9)]
		}
		train=shuffle(train)
		X_train, X_test = train_test_split(train, test_size=0.2)
		X=X_train[features]
		Y=X_train['Sales']
		print("validation data X and Y")
		grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", n_jobs=-1, cv=3)
		grid_result = grid_search.fit(X,Y)
		print("best parameters are:")
		print(grid_search.best_params_)
		test_pred=grid_result.predict(X_test[features])
		error = rmspe(test_pred, X_test['Sales'])
		print('error on using parameter tuned xgboost', error)
		

	def train_xgboost_model(self):
		print("predicting result with best parameters of xgboost")
		xgb_reg = XGBRegressor(max_depth = 14, learning_rate=0.1,min_child_weight = 11, subsample = 0.8, colsample_bytree = 0.7)
		xgb_reg=xgb_reg.fit(train[features], train['Sales'])
		test_pred=xgb_reg.predict(test[features])


	def save_in_csv(self):
		d = {'Id': test.Id, 'Sales': test_pred}
		output = pd.DataFrame(data=d)
		print("saving as xgboost_parameter_tuned.csv")
		output.to_csv('xgboost_parameter_tuned.csv',index=False)

	def make_pickel():
		print("making pikel file")
		xgboost_pkl = open(xgboost_pkl_filename, 'wb')
		pickle.dump(test_pred, xgboost_pkl)
		xgboost_pkl.close()

	def Linear_regression(self):
		lin_reg=LinearRegression()
		lin_reg=lin_reg.fit(X, Y)
		test_pred=lin_reg.predict(X_test[features])
		error = rmspe(test_pred, X_test['Sales'])
		print('error on using linear Regression', error)

	def Decision_tree_regressor(self):
		from sklearn.tree import DecisionTreeRegressor
		dtree_reg=DecisionTreeRegressor()
		dtree_reg=dtree_reg.fit(X, Y)
		test_pred=dtree_reg.predict(X_test[features])
		error = rmspe(test_pred, X_test['Sales'])
		print('error on using Decision Tree Regressor', error)

	def Random_forest_regressor(self):
		from sklearn.ensemble import RandomForestRegressor
		ranfor_reg=RandomForestRegressor()
		ranfor_reg=ranfor_reg.fit(X, Y)
		test_pred=ranfor_reg.predict(X_test[features])
		error = rmspe(test_pred, X_test['Sales'])
		print('error on using Random Forest Regressor', error)


if __name__ == "__main__":
	m=main()
	m.models_with_default_parameters()
