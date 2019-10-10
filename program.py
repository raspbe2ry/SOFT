import pandas as pd
from matplotlib import pylab as plt

train = pd.read_csv('E:/FAKULTET/MASTER/ISTRAZIVANJA/training/sales_train_v2.csv')
print ('number of shops: ', train['shop_id'].max())
print ('number of items: ', train['item_id'].max())
num_month = train['date_block_num'].max()
print ('number of month: ', num_month)
print ('size of train: ', train.shape)

train_clean = train.drop(labels = ['date', 'item_price'], axis = 1)
train_clean = train_clean.groupby(["item_id","shop_id","date_block_num"]).sum().reset_index()
train_clean = train_clean.rename(index=str, columns = {"item_cnt_day":"item_cnt_month"})
train_clean = train_clean[["item_id","shop_id","date_block_num","item_cnt_month"]]

shopId = 25
itemId = 3223
check = train_clean[["shop_id","item_id","date_block_num","item_cnt_month"]]
check = check.loc[check['shop_id'] == shopId]
check = check.loc[check['item_id'] == itemId]

plt.figure(figsize=(10,4))
plt.title('Check - Sales of Item' + str(itemId)+ ' at Shop ' + str(shopId))
plt.xlabel('Month')
plt.ylabel('Sales of Item '+ str(itemId)+' at Shop '+ str(shopId))
plt.plot(check["date_block_num"],check["item_cnt_month"])

month_list=[i for i in range(num_month+1)]
shop = []
for i in range(num_month+1):
    shop.append(shopId)
item = []
for i in range(num_month+1):
    item.append(itemId)
months_full = pd.DataFrame({'shop_id':shop, 'item_id':item,'date_block_num':month_list})

sales_33month = pd.merge(check, months_full, how='right', on=['shop_id','item_id','date_block_num'])
sales_33month = sales_33month.sort_values(by=['date_block_num'])
sales_33month.fillna(0.00,inplace=True)

month_steps = 6
list_of_months = []
for i in range(1,month_steps+1):
    sales_33month["T_" + str(i)] = sales_33month.item_cnt_month.shift(i)
    list_of_months.append("T_" + str(i))
sales_33month.fillna(0.0, inplace=True)
df = sales_33month[list_of_months+['shop_id','item_id','date_block_num','item_cnt_month']].reset_index()
df = df.drop(labels = ['index'], axis = 1)

train_df = df[0:31]
val_df = df[31:34]
x_val1 = val_df
mm = df[0:31]

x_full, y_full = df.drop(['shop_id','item_id','date_block_num','item_cnt_month'],axis=1),df.item_cnt_month
x_train,y_train = train_df.drop(['shop_id','item_id','date_block_num','item_cnt_month'],axis=1),train_df.item_cnt_month
x_val,y_val = val_df.drop(['shop_id','item_id','date_block_num','item_cnt_month'],axis=1),val_df.item_cnt_month
x_val2, y_val2 = mm.drop(['shop_id','item_id','date_block_num','item_cnt_month'],axis=1),mm.item_cnt_month

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
model_lstm = Sequential()
model_lstm.add(LSTM(32, input_shape=(month_steps,1)))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.fit_transform(x_val)
x_valid_scaled2 = scaler.fit_transform(x_val2)

print(x_train_scaled.shape[0], x_train_scaled.shape[1])

x_train_reshaped = x_train_scaled.reshape((x_train_scaled.shape[0], x_train_scaled.shape[1], 1))
x_val_resaped = x_valid_scaled.reshape((x_valid_scaled.shape[0], x_valid_scaled.shape[1], 1))
x_val_resaped2 = x_valid_scaled2.reshape((x_valid_scaled2.shape[0], x_valid_scaled2.shape[1], 1))

history = model_lstm.fit(x_train_reshaped, y_train, validation_data=(x_val_resaped, y_val),epochs=10000, batch_size=30, verbose=2, shuffle=False)
y_pre = model_lstm.predict(x_val_resaped)
y_pre2 = model_lstm.predict(x_val_resaped2)

fig, ax = plt.subplots()
ax.plot(df['date_block_num'], y_full, label='Actual')
ax.plot(mm['date_block_num'], y_pre2, label='PredictedTrain')
ax.plot(val_df['date_block_num'], y_pre, label='Predicted')
plt.title('LSTM Prediction vs Actual Sales for last 3 months')
plt.xlabel('Month')
plt.xticks(x_val1['date_block_num'])
plt.ylabel('Sales of Item '+str(itemId)+' at Shop '+str(shopId))
ax.legend()
plt.show()


