import  pandas  as  pd
import numpy  as np
train=pd.read_csv('/media/machine_learning/A80C461E0C45E7C01/all/numerai_datasets/numerai_training_data.csv')
print(train.head(3))
 
y=train['target']
train=train.drop(['target'],axis=1)
print(train.shape)
print(y.shape)
test=pd.read_csv('/media/machine_learning/A80C461E0C45E7C01/all/numerai_datasets/numerai_tournament_data.csv')
 

id=test['t_id']
test=test.drop(['t_id'],axis=1)
sub=pd.read_csv('/media/machine_learning/A80C461E0C45E7C01/all/numerai_datasets/example_predictions.csv')
trainvec=np.array(train)
testvec=np.array(test)
y=np.array(y)
from sklearn.linear_model  import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from bhtsne import tsne
temp=np.concatenate((trainvec,testvec))

temp=tsne(np.concatenate((trainvec,testvec)),3,10)
 
newtr=np.hstack((trainvec,temp[0:136573,:]))
newts=np.hstack((testvec,temp[136573:,:]))

#model = make_pipeline(PolynomialFeatures(2), LogisticRegression())
#model.fit(newtr,y)
model=LogisticRegression()
model.fit(newtr,y)
pred=model.predict_proba(newts)
train_pred5=model.predict_proba(newtr)
pd.DataFrame(train_pred5[:,1]).to_csv('/home/machine_learning/Downloads/sub6Train.csv',index=False)
#from sklearn.linear_model  import LogisticRegression
#from sklearn.pipeline import make_pipeline





#l=LogisticRegression()

#l.fit(newtr,y)
#pred=l.predict_proba(newts)
sub['probability']=pred[:,1]
sub.head(3)
print('prediction saved')
sub.to_csv('/home/machine_learning/Downloads/sub6.csv',index=False)


