import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split


def createTrainTest(data):

    print "Preparing Train and Test sets..."

    train = data[data['eval_set'] == 1].drop(['eval_set', 'user_id', 'product_id', 'order_id'], axis = 1)
    test =  data[data['eval_set'] == 2].drop(['eval_set', 'user_id', 'reordered'], axis = 1)

    prod =  data.drop(['eval_set', 'user_id', 'reordered'], axis = 1)

    X_train, X_eval, y_train, y_eval = train_test_split(
        train[train.columns.difference(['reordered'])], train['reordered'], test_size=0.1, random_state=2)

    return prod,X_train, X_eval, y_train, y_eval


def combine_order_product(z, df):
    product_bag = dict()
    thres_bag = dict()
    for row in df.itertuples():
        if row.reordered > z:
            try:
                product_bag[row.order_id] += ' ' + str(row.product_id)
                thres_bag[row.order_id] += ' ' + str(int(100 * row.reordered))
            except:
                product_bag[row.order_id] = str(row.product_id)
                thres_bag[row.order_id] = str(int(100 * row.reordered))

    for order in df.order_id:
        if order not in product_bag:
            product_bag[order] = ' '
            thres_bag[order] = ' '

    return product_bag, thres_bag


def runlgbm(prod,X_train, X_eval, y_train, y_eval):

    print "Training lgbm..."

    lgbm_train = lgb.Dataset(X_train, label=y_train)
    lgbm_eval = lgb.Dataset(X_eval, y_eval, reference = lgbm_train)

    params = {'task': 'train', 'boosting_type': 'gbdt',   'objective': 'binary', 'metric': {'binary_logloss', 'auc'},
        'num_iterations' : 1000, 'max_bin' : 100, 'num_leaves': 512, 'feature_fraction': 0.8,  'bagging_fraction': 0.95,
        'bagging_freq': 5, 'min_data_in_leaf' : 200, 'learning_rate' : 0.05}


    lgbm_model = lgb.train(params, lgbm_train, num_boost_round = 50, valid_sets = lgbm_eval, early_stopping_rounds=10)

    print "Applying model to all data - both train and test.."

    prod['reordered'] = lgbm_model.predict(prod[prod.columns.difference(
        ['order_id', 'product_id'])], num_iteration=lgbm_model.best_iteration)

    print "summarizing products and probabilities ..."

    t_thres = traintest.copy()
    i = 0

    for threshold in [0.17, 0.21, 0.25]:

        product_bag, thres_bag = combine_order_product(threshold, prod)
        product_temp = pd.DataFrame.from_dict(product_bag, orient='index')
        product_temp.reset_index(inplace=True)

        thres_temp = pd.DataFrame.from_dict(thres_bag, orient='index')
        thres_temp.reset_index(inplace=True)

        product_temp.columns = ['order_id', 'products']
        thres_temp.columns = ['order_id', 'thres']

        product_temp['list_prod'] = product_temp['products'].apply(lambda x: list(map(int, x.split())))
        thres_temp['list_thres'] = thres_temp['thres'].apply(lambda x: list(map(int, x.split())))

        cart_act = product_temp['products'].apply(lambda x: len(x.split())).mean()

        t_thres = t_thres.merge(product_temp, on='order_id', how='inner')
        t_thres = t_thres.merge(thres_temp, on='order_id', how='inner')
        t_thres.drop(['products', 'thres'], axis=1, inplace=True)

        t_thres['thres_avg'] = t_thres['list_thres'].apply(lambda x: 0.01 * np.mean(x) if x != [] else 0.).astype(np.float16)
        t_thres['thres_max'] = t_thres['list_thres'].apply(lambda x: 0.01 * np.max(x) if x != [] else 0.).astype(np.float16)
        t_thres['thres_min'] = t_thres['list_thres'].apply(lambda x: 0.01 * np.min(x) if x != [] else 0.).astype(np.float16)
        t_thres['f1'] = t_thres.apply(calc_f1, axis=1).astype(np.float32)

        F1 = t_thres['f1'].loc[t_thres['eval_set'] == 1].mean()

        t_thres = t_thres.rename(columns={'list_prod': 'prod' + str(i), 'f1': 'f1' + str(i), 'list_thres': 'threshold' + str(i),
                                'thres_avg': 'thres_avg' + str(i), 'thres_max': 'thres_max' + str(i), 'thres_min': 'thres_min' + str(i)})

        #print "threshold,F1,reorder_act,cart_act :  ", threshold, F1, reorder_act, cart_act
        print "Threshold,F1 :  ", threshold, F1
        i = i + 1

    t_thres['fm'] = t_thres[['f10', 'f11', 'f12']].idxmax(axis=1)
    t_thres['f1'] = t_thres[['f10', 'f11', 'f12']].max(axis=1)
    t_thres['fm'] = t_thres.fm.replace({'f10': 0, 'f11': 1, 'f12': 2}).astype(np.int32)

    print " f1 maximized ", t_thres['f1'].loc[t_thres['eval_set'] == 1].mean()
    f = open('f1.txt',"w")
    for i in t_thres['f1']:
        f.write(i+"\n")
        f.flush()
    f.close()
    return t_thres

def calc_f1(dt):

    y_t = dt.reorder_act
    y_p = dt.list_prod

    if y_t == '' and y_p == []:
        return 1.
    y_t = set(y_t)
    y_p = set(y_p)
    cs = len(y_t & y_p)

    if cs == 0:
        return 0.

    p = 1. * cs / len(y_p)
    r = 1. * cs / len(y_t)

    return 2 * p * r / (p + r)

if __name__=="__main__":

    data = pd.read_pickle('final_data')
    traintest = pd.read_pickle('traintest')
    reorder_act = np.load('pred_act.npy')
    prod,X_train, X_eval, y_train, y_eval = createTrainTest(data)
    t_thres = runlgbm(prod,X_train, X_eval, y_train, y_eval)
    print "saving data to pickle"
    t_thres.to_pickle('tt_data')