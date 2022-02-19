from sklearn.svm import SVC


def model_classif(train_space, val_space):
    model = SVC(kernel='rbf')
    model.fit(train_space[:,:-1], train_space[:,-1])
    val_pred = model.predict(val_space[:,:-1 ])
    return val_pred