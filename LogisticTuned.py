import pandas as pd
from sklearn.metrics import log_loss
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression


data = pd.read_csv('/media/machine_learning/A80C461E0C45E7C01/all/numerai_datasets/numerai_training_data.csv')

train, test = cross_validation.train_test_split(data,
                                                test_size=0.7,
                                                random_state=0)

features = ["feature1", "feature2", "feature3", "feature4", "feature5",
            "feature6", "feature7", "feature8", "feature9", "feature10",
            "feature11", "feature12", "feature13", "feature14", "feature15",
            "feature16", "feature17", "feature18", "feature19", "feature20",
            "feature21"]


penalties = [1, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01, 0.001, 0.0001]
best_penalty = 1.0
best_logloss = 1.0

for penalty in penalties:
    model = LogisticRegression(C=penalty, n_jobs=-1, random_state=0)
    model.fit(train[features], train['target'])
    test_predictions = model.predict_proba(test[features])

    logloss = log_loss(test['target'], test_predictions)
    print("Test Log Loss %f with penalty %f" % (logloss, penalty))

    if logloss <= best_logloss:
        best_penalty = penalty
        best_logloss = logloss

model = LogisticRegression(C=best_penalty, n_jobs=-1, random_state=0)
model.fit(data[features], data['target'])
test_predictions = model.predict_proba(data[features])
logloss = log_loss(data['target'], test_predictions)
print("Fully Trained Log Loss %f with penalty %f" % (logloss, best_penalty))

tournament_data = pd.read_csv('/media/machine_learning/A80C461E0C45E7C01/all/numerai_datasets/numerai_tournament_data.csv')
tournament_predictions = model.predict_proba(tournament_data[features])

result = tournament_data
result['probability'] = tournament_predictions[:,1]
sub6train=model.predict_proba(train[features])
pd.DataFrame(sub6train).to_csv('/home/machine_learning/Downloads/sub6train.csv',index=None)
result.to_csv("/home/machine_learning/Downloads/sub6.csv",
              columns=('t_id', 'probability'),
              index=None)
