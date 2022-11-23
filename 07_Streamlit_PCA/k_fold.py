from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
from model import k_fold
from sklearn.decomposition import PCA


def k_fold_evaluation(model, X_scale, y, n_splits, n_components):
    # KFold
    kf = KFold(n_splits=n_splits, random_state=None)
    f1_score_list = []

    for i in n_components:
        f1_score_fold = []
        pca = PCA(n_components=i)
        X_pca = pca.fit_transform(X_scale)
    
        for train_index , test_index in kf.split(X_pca):
            X_train , X_test = X_pca[train_index,:], X_pca[test_index,:]
            y_train , y_test = y[train_index] , y[test_index]

            model.fit(X_train, y_train)
            pred_values = model.predict(X_test)
            
            f1_score_fold.append(f1_score(pred_values , y_test, average='weighted'))
                    
        f1_score_list.append(sum(f1_score_fold) / n_splits)
    return f1_score_list