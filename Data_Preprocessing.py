import pandas as pd
import numpy as np
import gc


def initialize_df():

    print "Loading files started..."

    prior_order_products = pd.read_csv('/home/hari/Documents/Data Mining/Project/data/raw/order_products__prior.csv', dtype={'order_id': np.int32,'product_id': np.int16, 'reordered': np.int8, 'add_to_cart_order': np.int8})
    train_orders_products = pd.read_csv('/home/hari/Documents/Data Mining/Project/data/raw/order_products__train.csv', dtype={'order_id': np.int32,'product_id': np.int16, 'reordered': np.int8, 'add_to_cart_order': np.int8 })
    orders = pd.read_csv('/home/hari/Documents/Data Mining/Project/data/raw/orders.csv', dtype={'order_hour_of_day': np.int8,'order_number': np.int8, 'order_id': np.int32, 'user_id': np.int32,'order_dow': np.int8, 'days_since_prior_order': np.float16})
    products = pd.read_csv('/home/hari/Documents/Data Mining/Project/data/min/products.csv', dtype={'product_id': np.int16, 'aisle_id': np.int8, 'department_id': np.int8},usecols=['product_id', 'aisle_id', 'department_id'])

    orders.eval_set = orders.eval_set.replace({'prior': 0, 'train': 1, 'test':2}).astype(np.int32)
    orders.days_since_prior_order = orders.days_since_prior_order.fillna(30).astype(np.int32)

    print "done loading input files"
    return prior_order_products,train_orders_products,orders,products


# Merging order_prior_products and orders
def merge_df(orders,prior_order_products,train_orders_products):

    orders_products = orders.merge(prior_order_products, how = 'inner', on = 'order_id')
    train_orders_products = train_orders_products.merge(orders[['user_id','order_id']], left_on = 'order_id', right_on = 'order_id', how = 'inner')

    return orders_products,train_orders_products


def create_data(orders,orders_products,products):

    prod_sort = orders_products.sort_values(['user_id', 'order_number', 'product_id'], ascending=True)
    prod_sort['p_time'] = prod_sort.groupby(['user_id', 'product_id']).cumcount()+1

    #Accessing 1st and 2nd time order products
    order_1 = prod_sort[prod_sort['p_time'] == 1].groupby('product_id').size().to_frame('prod_first_orders')
    order_2 = prod_sort[prod_sort['p_time'] == 2].groupby('product_id').size().to_frame('prod_second_orders')

    order_1['prod_orders'] = prod_sort.groupby('product_id')['product_id'].size()
    order_1['prod_reorders'] = prod_sort.groupby('product_id')['reordered'].sum()
    order_2 = order_2.reset_index().merge(order_1.reset_index())
    order_2['prod_reorder_prob'] = order_2['prod_second_orders']/order_2['prod_first_orders']
    order_2['prod_reorder_ratio'] = order_2['prod_reorders']/order_2['prod_orders']
    p_prd = order_2[['product_id', 'prod_orders','prod_reorder_prob', 'prod_reorder_ratio']]

    print "Test2"
    # extracting user info
    users = orders[orders['eval_set'] == 0].groupby(['user_id'])['order_number'].max().to_frame('user_orders')
    users['user_period'] = orders[orders['eval_set'] == 0].groupby(['user_id'])['days_since_prior_order'].sum()
    users['user_md'] = orders[orders['eval_set'] == 0].groupby(['user_id'])['days_since_prior_order'].mean()

    # combining user and order
    user = orders_products.groupby('user_id').size().to_frame('user_tp')
    user['equal'] = orders_products[orders_products['reordered'] == 1].groupby('user_id')['product_id'].size()
    user['greater'] = orders_products[orders_products['order_number'] > 1].groupby('user_id')['product_id'].size()
    user['user_reorder_ratio'] = user['equal'] / user['greater']
    user.drop(['equal', 'greater'], axis = 1, inplace = True)
    user['user_dp'] = orders_products.groupby(['user_id'])['product_id'].nunique()

    users = users.reset_index().merge(user.reset_index())
    users['user_bsize'] = users['user_tp'] / users['user_orders']

    user = orders[orders['eval_set'] != 0]
    user = user[['user_id', 'order_id', 'eval_set', 'days_since_prior_order']]
    users = users.merge(user)


    print "Test3"
    # merging orders and products and grouping by user and product
    data = orders_products.groupby(['user_id', 'product_id']).size().to_frame('user_porder')
    data['user_pforder'] = orders_products.groupby(['user_id', 'product_id'])['order_number'].min()
    data['user_lforder'] = orders_products.groupby(['user_id', 'product_id'])['order_number'].max()
    data['user_pavg_pos'] = orders_products.groupby(['user_id', 'product_id'])['add_to_cart_order'].mean()
    data = data.reset_index()

    #merging previous data with users
    data = data.merge(p_prd, on = 'product_id')
    data = data.merge(users, on = 'user_id')

    data['user_po_rate'] = data['user_porder'] / data['user_orders']
    data['up_orders_since_last_order'] = data['user_orders'] - data['user_lforder']
    data = data.merge(train_orders_products[['user_id', 'product_id', 'reordered']], how = 'left', on = ['user_id', 'product_id'])
    data = data.merge(products, on = 'product_id')

    return data

def generate_traintest(orders,train_orders_products):

    print " Train test data..."

    train_orders_products = train_orders_products[train_orders_products['reordered']==1].drop('reordered',axis=1)
    orders.set_index('order_id', drop=False, inplace=True)
    train=orders[['order_id','eval_set']].loc[orders['eval_set']==1]
    train['reorder_act'] = train_orders_products.groupby('order_id').aggregate({'product_id':lambda x: list(x)})
    train['reorder_act']=train['reorder_act'].fillna('')

    pred_act = train['reorder_act'].apply(lambda x: len(x)).mean()

    test=orders[['order_id','eval_set']].loc[orders['eval_set']==2]
    test['reorder_act']=' '
    train_test=pd.concat([train,test])
    train_test.set_index('order_id', drop=False, inplace=True)

    return train_test,orders,pred_act

def format_data(data):

    print "Test inside format data"

    data = data.astype(dtype= {'user_id' : np.int32,
                               'product_id'  : np.int16,
                               'user_porder'  : np.int8,
                               'user_pforder' : np.int8,
                               'user_lforder' : np.int8,
                               'user_pavg_pos' : np.int8,
                               'prod_orders' : np.int16,
                               'prod_reorder_prob' : np.float16,
                               'prod_reorder_ratio' : np.float16,
                               'user_orders' : np.int32,
                               'user_period' : np.int8,
                               'user_md' : np.int8,
                               'user_tp' : np.int8,
                               'user_reorder_ratio' : np.float16,
                               'user_dp' : np.int8,
                               'user_bsize' : np.int8,
                               'order_id'  : np.int32,
                               'eval_set' : np.int8,
                               'days_since_prior_order' : np.int8,
                               'user_po_rate' : np.float16,
                               'up_orders_since_last_order':np.int8,
                               'aisle_id': np.int8,
                               'department_id': np.int8})
    data['reordered'].fillna(0, inplace=True)
    data['reordered']=data['reordered'].astype(np.int32)
    print "data:",len(data)
    print "traintest len:",len(train_test)

    return data

if __name__=="__main__":

    prior_order_products,train_orders_products,orders,products = initialize_df();
    orders_products,train_orders_products = merge_df(orders,prior_order_products,train_orders_products);
    data = create_data(orders,orders_products,products)
    train_test,orders,pred_act = generate_traintest(orders,train_orders_products)
    data = format_data(data)

    data.to_pickle('final_data')
    train_test.to_pickle('traintest')
    np.save('pred_act', pred_act)