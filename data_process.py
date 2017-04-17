import csv
data_no_cyc = [];
data_with_cyc = [];
with open('test1.csv', 'rb') as csvfile:
	data_pre = csv.reader(csvfile)
	for row in data_pre:
		if row[4] != 'NULL':
			data_with_cyc.append(row)
		elif row[4] == 'NULL':
			del row[4:12]
			data_no_cyc.append(row)

data_with_cyc_after = [];
data_no_cyc_after = [];

for i in range(len(data_with_cyc) - 9):
	if  data_with_cyc[i][0] == data_with_cyc[i + 9][0]:
		for j in range(10):
			data_with_cyc_after.append(data_with_cyc[i + j])

for i in range(len(data_no_cyc) - 9):
	if  data_no_cyc[i][0] == data_no_cyc[i + 9][0]:
		for j in range(10):
			data_no_cyc_after.append(data_no_cyc[i + j])

with open("output1.csv", "wb") as f1:
    writer = csv.writer(f1)
    writer.writerows(data_with_cyc_after)

with open("output2.csv", "wb") as f2:
    writer = csv.writer(f2)
    writer.writerows(data_no_cyc_after)


