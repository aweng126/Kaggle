import os
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

TITANIC_PATH = os.path.join("../datasets","titanic")
def load_titanic_data(filename,titanic_path=TITANIC_PATH):
	csv_path = os.path.join(titanic_path,filename)
	return pd.read_csv(csv_path)


class DataFrameSelector(BaseEstimator,TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names = attribute_names
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        return X[self.attribute_names]

class MostFrequentImputer(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],index=X.columns)
        return self
    def transform(self,X,y=None):
        return X.fillna(self.most_frequent_)


def gen_submission(filename=None,X=None,Y=None):

	#字典中的key值即为csv中列名
	dataframe = pd.DataFrame({'PassengerId':X,'Survived':Y})

	#将DataFrame存储为csv,index表示是否显示行名，default=True
	dataframe.to_csv(filename,index=False,sep=',')


def main():
	# 加载数据
	train_set = load_titanic_data("train.csv")
	print(train_set.head(1))
	test_set = load_titanic_data("test.csv")
	#print(train_set.info())

	# 清洗数据
	num_pipeline=Pipeline([
    	("select_numeric",DataFrameSelector(["Age","SibSp","Parch","Fare"])),
   		("imputer",SimpleImputer(strategy="median")),
    ])
	num_pipeline.fit_transform(train_set)

	cat_pipeline = Pipeline([
	    ("select_cat",DataFrameSelector(["Pclass","Sex","Embarked"])),
	    ("imputer",MostFrequentImputer()),
	    ("cat_encoder",OneHotEncoder(sparse=False)),
	])
	cat_pipeline.fit_transform(train_set)


	preprocess_pipeline = FeatureUnion(transformer_list=[
	    ("num_pipeline",num_pipeline),
	    ("cat_pipeline",cat_pipeline),
	])

	X_train = preprocess_pipeline.fit_transform(train_set)
	print(X_train[0])
	y_train = train_set["Survived"]

	# 支持向量机算法	
	svm_clf = SVC(gamma="auto")
	svm_clf.fit(X_train,y_train)
	X_test = preprocess_pipeline.fit_transform(test_set)
	y_pred = svm_clf.predict(X_test)

	svm_scores = cross_val_score(svm_clf,X_train,y_train,cv=10)
	print("SVM: "+str(svm_scores.mean()))

	#gen_submission("gender_submission.csv",test_set["PassengerId"],y_pred)

	# 随机森林算法
	forest_clf = RandomForestClassifier(n_estimators=100,random_state=42)
	forest_clf.fit(X_train,y_train)
	forest_pred = forest_clf.predict(X_test)
	#10折交叉验证
	forest_scores = cross_val_score(forest_clf,X_train,y_train,cv=10)
	print("random forest: "+str(forest_scores.mean()))

    #保存利用random forest的预测结果
	#gen_submission("gender_submission.csv",test_set["PassengerId"],forest_pred)


	#adboost 算法
	adaboost_clf = AdaBoostClassifier(n_estimators=100)
	adaboost_clf.fit(X_train,y_train)
	adaboost_pred = adaboost_clf.predict(X_test)
	#10折交叉验证
	adaboost_scores = cross_val_score(adaboost_clf,X_train,y_train,cv=10)
	print("adaboost: "+str(adaboost_scores.mean()))

	gen_submission("gender_submission.csv",test_set["PassengerId"],adaboost_pred)


if __name__ == '__main__':
	main()