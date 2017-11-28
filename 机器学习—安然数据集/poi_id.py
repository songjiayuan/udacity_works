#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import RandomizedPCA
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
target_label=['poi']
v={'salary':0, 'to_messages':0, 'email_address':0,'deferral_payments':0, 'total_payments':0, 'exercised_stock_options':0, 'bonus':0, 'restricted_stock':0, 'shared_receipt_with_poi':0, 'restricted_stock_deferred':0, 'total_stock_value':0, 'expenses':0, 'loan_advances':0, 'from_messages':0, 'other':0, 'from_this_person_to_poi':0, 'director_fees':0, 'deferred_income':0, 'long_term_incentive':0, 'from_poi_to_this_person':0}

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)  #json格式的数据
    #print(data_dict)
    total_data_point=0
    features=[]
    valid_data={}
    valid_data_number=0
    unvalue_data=0
    poi_number=0
    number=0
    feature_number=0
#统计数据
for name,value in data_dict.items():
    for valu,val in value.items():
        if val=='NaN':
            v[valu]+=1
    #print v 
	number+=1
	total_data_point+=len(value)   #数据总数
    
	for feature in value:
		if feature not in features and feature!='poi' and feature !='email_address':
			features.append(feature)
			feature_number=len(features)
#print features   #安然数据集中的特征
		if value[feature]!='NaN':
			valid_data_number+=1
		if value[feature]=='NaN':
			unvalue_data+=1
		if feature=='poi' and value[feature]==True:
			poi_number+=1
	valid_data[name]=valid_data_number
#print valid_data #这就是一些有效的数据他们的顺序

#print total_data_point,valid_data,unvalue_data,poi_number,number,feature_number,features
#print v
#每个特征中NaN的数量，'salary': 51, 'to_messages': 60, 'deferral_payments': 107,
 #'total_payments': 21, 'loan_advances': 142, 'bonus': 64, 'restricted_stock': 36
 # 'restricted_stock_deferred': 128, 'total_stock_value': 20, 'shared_receipt_with_poi': 60,
 #  'long_term_incentive': 80, 'exercised_stock_options': 44, 'from_messages': 60, 'other': 53, 
 #'from_poi_to_this_person': 60, 'from_this_person_to_poi': 60, 'deferred_income': 97, 'expenses': 51, 'email_address': 35, 'director_fees': 129
#所以说：一共有3066个数据，其中1708个有效，1358个为NaN无效，共146个人其中18个嫌疑人
#所以这个数据集有很多缺失值，也就说明accuracy并不是很好的评估指标，选择precision和recall更好一些。
#在交叉验证的时候，因为数据的不平衡性，我们会选用Stratified Shuffle Split的方式将数据分为验证集和测试集。
#数据样本比较少，因此我们可以使用GridSearchCV来进行参数调整，如果较大的数据则会花费较长的时间，可以考虑使用RandomizedSearchCV


### Task 2: Remove outliers
#处理异常值，无效的名字，无效的特征
out_names=[]
for name in valid_data:
	if len(name.split())>4:
		out_names.append(name)
	if len(name.split())<2:
		out_names.append(name)
#print out_names
#print len(data_dict)
#restricted_stock_deferred，director_fees，存在大量缺失值，为无效特征，从特征中删除这两项


#除去异常值
for name in valid_data:
	if name in out_names:
		data_dict.pop(name,0)
features=['salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 
'restricted_stock', 'shared_receipt_with_poi',  'total_stock_value', 'expenses', 'loan_advances', 
'from_messages', 'other', 'from_this_person_to_poi', 'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']



### Task 3: Create new feature(s)新特征
new_features = ["total_asset", "fraction_of_messages_with_poi"]  
asset = ["salary", "bonus", "total_stock_value", "exercised_stock_options"]
messages = ["to_messages", "from_messages", "from_poi_to_this_person", "from_this_person_to_poi"]

for name, value in data_dict.items():
    valid_asset_value = True
    valid_message_value = True
    for key in asset:
        if value[key] == "NaN":
            valid_asset_value = False
            break
    for key in messages:
        if value[key] == "NaN":
            valid_message_value = False
            break
    if valid_asset_value:
        ### sum total asset if data are valid
        value[new_features[0]] = sum([value[key] for key in asset]) 
    else:
        ### assign value to NaN if having invalid data
        value[new_features[0]]  = "NaN"
    if valid_message_value:
        ### calculate fraction of message interacting with poi
        all_messages = value["to_messages"] + value["from_messages"]
        messages_with_poi = value["from_poi_to_this_person"] + value["from_this_person_to_poi"]
        value[new_features[1]] = float(messages_with_poi)/all_messages
    else:
        ### assign value to NaN if having invalid data
        value[new_features[1]] = "NaN"

feature_list=target_label+features+new_features
#对比加入和不加入新特征算法的性能，明显，加入新特征性能更好一点
#Accuracy: 0.8372093   Precision: 0.55753  Recall: 0.67682
#Accuracy: 0.81395  Precision: 0.486111  Recall: 0.41666



### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
# data = featureFormat(my_dataset, feature_list, sort_keys = True)
# labels, features = targetFeatureSplit(data)

from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()

data = featureFormat(data_dict, feature_list)
labels,features = targetFeatureSplit(data)
features = scaler.fit_transform(features)
#print features
from sklearn.feature_selection import SelectKBest
#用k-best进行特征选择
k_best = SelectKBest(k=10)
k_best.fit(features, labels)
scores = k_best.scores_
feature_and_score = sorted(zip(feature_list[1:], scores), key = lambda l: l[1],\
     reverse = True)
feature_list = target_label + [feature for (feature, score) in \
    feature_and_score][:10]
print "Top 10 features and scores:\n", feature_and_score[:10], "\n"
print "My features:\n", feature_list, "\n"
#经过迭代，选择最优的k值
# k=10
# (0.86046511627906963, 0.68650793650793651, 0.74736842105263168)
# Accuracy: 0.83916   Precision: 0.35294  Recall: 0.33333 
# (0.86046511627906963, 0.62896825396825373, 0.77083333333333326)
# Accuracy: 0.71329   Precision: 0.21951  Recall: 0.50000 

data = featureFormat(data_dict, feature_list)
labels, features = targetFeatureSplit(data)
features = scaler.fit_transform(features)
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn import svm
s_clf = svm.SVC(kernel="rbf")

from sklearn import linear_model
g_clf = linear_model.SGDClassifier(class_weight = "auto")

### Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()

### Random Forests
from sklearn.ensemble import RandomForestClassifier
r_clf = RandomForestClassifier()



trials=500
is_pca=True
n=5


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)

def clf_evaluation(clf, features, labels, trials, is_pca, n):
  
    score = []
    precision = []
    recall = []
    for trial in range(trials): 
        features_train, features_test, labels_train, labels_test = \
            train_test_split(features, labels, test_size = 0.3, random_state = 42)
        if is_pca:
            pca = RandomizedPCA(n_components = n, whiten=True).fit(features_train)
            features_train = pca.transform(features_train)
            features_test = pca.transform(features_test)
        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        #print pred
        score.append(accuracy_score(pred, labels_test))
        #print score
        precision.append(precision_score(pred, labels_test,average=None))
        #print precision
        recall.append(recall_score(pred, labels_test,average=None))
        #Zprint recall
    print (np.mean(score), np.mean(precision), np.mean(recall))
#通过调整参数，观察算法的性能
#朴素贝叶斯并不具备可调整的参数
#选择linear_model进行参数调整




#clf_evaluation(s_clf, features, labels, trials, is_pca, n)
clf_evaluation(nb_clf, features, labels, trials, is_pca, n)
#clf_evaluation(g_clf, features, labels, trials, is_pca, n)
#他的各项性能相比朴素贝叶斯还是挺低的
from sklearn import grid_search
trials = 1
parameters = {"loss":("hinge", "log", "modified_huber", "perceptron", "epsilon_insensitive", 
 "huber"), "n_iter":[1,5,10,20, 50, 100, 200, 500],"alpha": [1e-15, 1e-10, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}
g_clf = grid_search.GridSearchCV(g_clf, parameters)
print clf_evaluation(g_clf, features, labels, trials, is_pca, n), "\n"
g_clf =  g_clf.best_estimator_
#参数调整之后比之前好，但还是不如朴素贝叶斯
#Accuracy: 0.46154   Precision: 0.09589  Recall: 0.38889


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results
dump_classifier_and_data(nb_clf, my_dataset, feature_list)
#测试
print 'Tester Classification report'
test_classifier(nb_clf,my_dataset,feature_list)




