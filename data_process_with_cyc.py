
def input_data(steps):
    batch_x = data_with_cyc_after[(steps - 1) * batch_size:steps * batch_size]
    batch_y = data_with_cyc_after_output[steps - 1]
    return batch_x, batch_y # batch_x 128*28*28 batch_y 128*10


import csv
data_with_cyc = [];
data_with_cyc_str = [];
batch_size = 20;

with open('test2.csv', 'rb') as csvfile:
	data_pre = csv.reader(csvfile, delimiter=',', quotechar='"')

	for row in data_pre:
		if row[4] != 'NULL':
			del row[26:36]
			del row[17:24]
			del row[9:14]
			del row[1:6]
			data_with_cyc_str.append(row)



for row in data_with_cyc_str:
	float_row = []

	for col in row:
		float_row.append( float(col))
	data_with_cyc.append(float_row)

data_with_cyc_after_temp = []
data_with_cyc_after_output_temp = []

for i in range(len(data_with_cyc) - 9):
	if  data_with_cyc[i][0] == data_with_cyc[i + 9][0]:
		for j in range(10):
			data_with_cyc_after_output_temp.append(list(data_with_cyc[i+j][-2:]))
			temp = list(data_with_cyc[i + j])
			del temp[0]
			del temp[-2:]
			data_with_cyc_after_temp.append(temp)
#print data_no_cyc_after



# data_with_cyc_after = [];
# data_with_cyc_after_output = [];
# for i in range(len(data_with_cyc_after_temp))[0::10]:
# 	data_with_cyc_after.append(data_with_cyc_after_temp[i:i+10])

# num_batch = len(data_with_cyc_after_output_temp)/batch_size


# for k in range(num_batch):
# 	temp1 = []
# 	for j in range(10):
# 		temp = []
# 		for i in range(batch_size*10)[0::10]:
# 			temp.append(data_with_cyc_after_output_temp[j])
# 		temp1.append(temp)
# 	data_with_cyc_after_output.append(temp1)
# [batch_x,batch_y] = input_data(1)

print len(data_with_cyc_after_temp)
#data_with_cyc_after[(step - 1) * 5:step * 5]

