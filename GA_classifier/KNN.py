from sklearn.neighbors import KNeighborsClassifier


def model_classif(train_space, val_space):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(train_space[:,:-1], train_space[:,-1])
    val_pred = model.predict(val_space[:,:-1 ])
    return val_pred