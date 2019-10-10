import pandas as pd
from matplotlib import pylab as plt

train = pd.read_csv('E:/FAKULTET/MASTER/ISTRAZIVANJA/training/final.csv')
num_month = train['date_block_num'].max()
print(num_month)

check = train[["shop_id","item_category_id","date_block_num","item_cnt_month"]]
check = check.loc[check['shop_id'] == 5]
check = check.loc[check['item_category_id'] == 6]

plt.figure(figsize=(10,4))
plt.title('Check - Sales of Item Category 6 at Shop 5')
plt.xlabel('Month')
plt.ylabel('Sales of Item Category 6 at Shop 5')
plt.plot(check["date_block_num"],check["item_cnt_month"])

month_list=[i for i in range(num_month+1)]
shop = []
for i in range(num_month+1):
    shop.append(5)
category = []
for i in range(num_month+1):
    category.append(6)
months_full = pd.DataFrame({'shop_id':shop, 'item_category_id':category,'date_block_num':month_list})

sales_33month = pd.merge(check, months_full, how='right', on=['shop_id','item_category_id','date_block_num'])
sales_33month = sales_33month.sort_values(by=['date_block_num'])
sales_33month.fillna(0.00,inplace=True)

month_steps = 12
list_of_months = []
for i in range(1,month_steps+1):
    sales_33month["T_" + str(i)] = sales_33month.item_cnt_month.shift(i)
    list_of_months.append("T_" + str(i))
sales_33month.fillna(0.0, inplace=True)

original = sales_33month[-3:]
df = sales_33month[list_of_months+['item_cnt_month']].reset_index()
df = df.drop(labels = ['index'], axis = 1)

train_df = df[:-3]
val_df = df[-3:]
x_train,y_train = train_df.drop(["item_cnt_month"],axis=1),train_df.item_cnt_month
x_val,y_val = val_df.drop(["item_cnt_month"],axis=1),val_df.item_cnt_month

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
model_lstm = Sequential()
model_lstm.add(LSTM(15, input_shape=(month_steps, 1)))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.fit_transform(x_val)

x_train_reshaped = x_train_scaled.reshape((x_train_scaled.shape[0], x_train_scaled.shape[1], 1))
x_val_resaped = x_valid_scaled.reshape((x_valid_scaled.shape[0], x_valid_scaled.shape[1], 1))

history = model_lstm.fit(x_train_reshaped, y_train, validation_data=(x_val_resaped, y_val),epochs=15000, batch_size=30, verbose=2, shuffle=False)
y_pre = model_lstm.predict(x_val_resaped)


fig, ax = plt.subplots()
ax.plot(original['date_block_num'], y_val, label='Actual')
ax.plot(original['date_block_num'], y_pre, label='Predicted')
plt.title('LSTM Prediction vs Actual Sales for last 3 months')
plt.xlabel('Month')
plt.xticks(original['date_block_num'])
plt.ylabel('Sales of Item Category 6 at Shop 5')
ax.legend()
plt.show()

