# -*- coding: utf-8 -*-
import pandas as pd

print("Reading the raw data file")
raw_data = pd.read_csv(("C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\NASDAQ_62_with_indicators.CSV"),header = 0, sep=",")

print(len(raw_data))

raw_data = raw_data[['Equity','Date','Open','High','Low','Close','Volume','Significant_1D_Rise1.0STD']]

positive_samples = raw_data[raw_data['Significant_1D_Rise1.0STD']==1.0]
print(len(positive_samples))
negative_samples = raw_data[raw_data['Significant_1D_Rise1.0STD']==0.0]
print(len(negative_samples))

pos_sampled = positive_samples.sample(n=2000)
neg_sampled = negative_samples.sample(n=2000)
sampled_data = pos_sampled.append(neg_sampled,ignore_index=True)
sampled_data = sampled_data.iloc[np.random.permutation(len(sampled_data))]
print(len(pos_sampled))
print(len(neg_sampled))

dates_for_rnn = list(pos_sampled['Date']) + list(neg_sampled['Date'])

print len(dates_for_rnn)

count = 0
for date in dates_for_rnn:
	row = (raw_data[raw_data['Date']==date]).index.get_values()[0]
	if row>10:
		features_data = raw_data.iloc[row-10:row,1:6]
		labels_data = raw_data.iloc[row-10:row,6:]
		features_data.to_csv(('/Users/ROBIN/Desktop/MS_and_I_Res/HQ/Code/RNN_Java/myInput_'+str(count)+'.csv'),sep = ',',header=False,index=None)
		(labels_data.tail(1)).to_csv(('/Users/ROBIN/Desktop/MS_and_I_Res/HQ/Code/RNN_Java/myLabels_'+str(count)+'.csv'),sep = ',',header=False,index=None)
		count += 1

count = 0
class_1,class_0 = 0,0
for row,value in sampled_data.iterrows():
    equity_name = value['Equity']
    date = value['Date']
    print(equity_name,date)
    subset = raw_data[(raw_data['Equity']==equity_name)]
    subset['Significant_1D_Rise1.0STD'] = subset['Significant_1D_Rise1.0STD'].apply(int)
    row_index = (subset[subset['Date']==date]).index.get_values()[0]
    #print(row_index)    
    #if count > 20:
    #    break
    #print(count)
    if row_index>10 and (row_index-9) >= subset.index.get_values()[0]:
        features_data =  subset.ix[[i for i in range(row_index-9,row_index+1)],[2,3,4,5,6]]
        #print(features_data)
        labels_data = subset.ix[[row_index],[7]]
        class_value = (subset.ix[[row_index],[7]]).iloc[0]['Significant_1D_Rise1.0STD']
        #print(labels_data)
        features_data.to_csv(("C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\Recurrent_NN_model\\Data_in_RNN_form\\myInput_"+ str(count)+'.csv'),sep = ',',header=False,index=None)
        labels_data.to_csv(("C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\Recurrent_NN_model\\Data_in_RNN_form\\myLabels_"+ str(count)+'.csv'),sep = ',',header=False,index=None)
        count+=1
        if (class_value == 1):
            class_1 += 1
        elif (class_value == 0):
            class_0 += 1
        print (count,class_1,class_0)
        
        