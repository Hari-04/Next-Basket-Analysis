from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from sklearn.model_selection import train_test_split


def GBClassifier(data):

    print "starting level 2..."
    X = data[[ 'thres_avg0', 'thres_max0','thres_min0', 'thres_avg1', 'thres_max1', 'thres_min1', 'thres_avg2', 'thres_max2', 'thres_min2']].loc[data['eval_set']==1]
    y = data['fm'].loc[data['eval_set']==1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    clf = GradientBoostingClassifier().fit(X_train, y_train)
    print "GB Accuracy on training set: {:.2f}" .format(clf.score(X_train, y_train))
    print "GB Accuracy on test set: {:.2f}" .format(clf.score(X_test, y_test))

    final = data[['order_id','prod0','prod1','prod2','thres_avg0']].loc[data['eval_set'] == 2]
    df_test = data[[ 'thres_avg0', 'thres_max0','thres_min0', 'thres_avg1', 'thres_max1', 'thres_min1', 'thres_avg2', 'thres_max2', 'thres_min2']].loc[data['eval_set']==2]
    final['fit'] = clf.predict(df_test)
    final['best'] = final.apply(lambda row: row['prod0'] if row['fit'] == 0 else ( row['prod1'] if row['fit'] == 1 else  row['prod2'] )  , axis=1)
    final['products'] = final.apply(fill_products, axis=1)

    return final

def fill_products(x):

    p_best = x.best
    thres_avg = x.thres_avg0

    if p_best == []:
        return 'None'
    if thres_avg < 0.5:
        if len(p_best) == 1:
            return str(p_best[0]) + ' None'
        if len(p_best) == 2:
            return str(p_best[0]) + ' ' + str(p_best[1]) + ' None'
    return ' '.join(str(i) for i in p_best)

if __name__=="__main__":

    data = pd.read_pickle('tt_data')
    final_result = GBClassifier(data)
    final_result[['order_id', 'products']].to_csv('predictions.csv', index=False)

