from datetime import datetime

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import np_utils
from sklearn.metrics import log_loss
from sklearn import cross_validation

startTime = datetime.now()
print("Start time: %s" % str(startTime))

# fix random seed for reproducibility
# seed = 7
# np.random.seed(seed)

print('Loading training data')
data = pd.read_csv('/media/machine_learning/A80C461E0C45E7C01/all/numerai_datasets/numerai_tournament_data.csv')

train, test = cross_validation.train_test_split(data, test_size = 0.2)

features = ["feature1","feature2","feature3","feature4","feature5","feature6","feature7","feature8",
            "feature9","feature10","feature11","feature12","feature13","feature14","feature15",
            "feature16","feature17","feature18","feature19","feature20","feature21"]


def create_model():
    model = Sequential()
    model.add(Dense(15, input_dim=21, init='normal', activation='sigmoid'))
    model.add(Dense(10, init='normal', activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(5, activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = create_model()

print('Training network')

y_train = np_utils.to_categorical(data['target'])

model.fit(data[features].as_matrix(), y_train, batch_size=250, nb_epoch=20, verbose=2)
test_predictions = model.predict_proba(data[features].as_matrix())
logloss = log_loss(data['target'], test_predictions)
print("Fully Trained Log Loss %f" % logloss)


print('Loading tournament data')
tournament_data = pd.read_csv('/media/machine_learning/A80C461E0C45E7C01/all/numerai_datasets/numerai_tournament_data.csv')
tournament_predictions = model.predict_proba(tournament_data[features].as_matrix())
pd.DataFrame(test_predictions).to_csv('/home/machine_learning/Downloads/sub7train.csv',index=False)

result = tournament_data
result['probability'] = tournament_predictions[:,1]

result.to_csv("/home/machine_learning/Downloads/sub7.csv", columns= ('t_id', 'probability'), index=None)

hours, remainder = divmod((datetime.now() - startTime).seconds, 3600)
minutes, seconds = divmod(remainder, 60)
print('Took %s:%s:%s' % (hours, minutes, seconds))
