import pandas as pd
import matplotlib.pyplot as plt
import pickle
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
data =pd.read_csv('breast-cancer-wisconsin.data',
names=[
       'Sample_code_number',
       'Clump_Thickness',
       'Uniformity_of_Cell_Size',
       'Uniformity_of_Cell_Shape',
       'Marginal_Adhesion',
       'Single_Epithelial_Cell_Size',
       'Bare_Nuclei',
       'Bland_Chromatin',
       'Normal_Nucleoli',
       'Mitoses',
       'Class'
       ])

#print(data.head())
#print(data.dtypes)
def is_non_numeric(x):
    return not x.isnumeric()

mask=data['Bare_Nuclei'].apply(is_non_numeric)
data_numeric=data[~mask]
data_numeric.head()
#print(len(data))
#print(len(data_numeric))
data_numeric['Bare_Nuclei']=data_numeric['Bare_Nuclei'].astype('int64')
#print(data_numeric.dtypes)
data_input=data_numeric.drop(columns=['Sample_code_number','Class'])
data_output=data_numeric['Class']
#print(data_output.head())
data_output= data_output.replace({2:0,4:1})
#print(data_output.unique())
#print(data_output.head)
x, x_test , y, y_test=train_test_split(data_input,data_output,test_size=0.33,random_state=2)
x_train ,x_val,y_train , y_val =train_test_split(x,y,test_size=0.33, random_state=2)
#print(x_train.shape , x_val.shape)
#print(y_train.shape, y_val.shape)

max_depth_values =[1,2,3,4,5,6,7,8]
train_accuracy_values=[]
val_accuracy_values=[]

for max_depth_val in max_depth_values:
    model=DecisionTreeClassifier(max_depth=max_depth_val,random_state=2)
    model.fit(x_train,y_train)
    y_pred_train=model.predict(x_train)
    y_pred_val=model.predict(x_val)
    acc_train = accuracy_score(y_train,y_pred_train)
    acc_val = accuracy_score(y_val,y_pred_val)
    train_accuracy_values.append(acc_train)
    val_accuracy_values.append(acc_val)

plt.plot(max_depth_values,train_accuracy_values,label='acc train')
plt.plot(max_depth_values,val_accuracy_values,label='val train')
plt.legend()
plt.grid(axis='both')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.title('Effect of max_depth on accuracy')
plt.show()

model_best =DecisionTreeClassifier(max_depth=3,random_state=2)
model_best.fit(x_train, y_train)
y_pred_test=model_best.predict(x_test)
print(accuracy_score(y_test, y_pred_test))

with open('saved-model.pickle','wb') as f:
    pickle.dump(model_best, f)
    
with open('saved-model.pickle','rb') as f:
    loaded_model=pickle.load(f)


plt.Figure(figsize=(25,20))
tree.plot_tree(model_best,
               feature_names=[
                   'Clump_Thickness',
                   'Uniformity_of_Cell_Size',
                   'Uniformity_of_Cell_Shape',
                   'Marginal_Adhesion',
                   'Single_Epithelial_Cell_Size',
                   'Bare_Nuclei',
                   'Bland_Chromatin',
                   'Normal_Nucleoli',
                   'Mitoses'
                   ],
               class_names=['benign','maligent'],
               filled=True)

plt.show()

#print(model_best.feature_importances_)
feature_names=[
    'Clump_Thickness',
    'Uniformity_of_Cell_Size',
    'Uniformity_of_Cell_Shape',
    'Marginal_Adhesion',
    'Single_Epithelial_Cell_Size',
    'Bare_Nuclei',
    'Bland_Chromatin',
    'Normal_Nucleoli',
    'Mitoses'
    ]

plt.bar(feature_names,model_best.feature_importances_)
plt.xlabel('features')
plt.xticks(rotation=90)
plt.ylabel('importance')
plt.title('Feature importances')
plt.show()