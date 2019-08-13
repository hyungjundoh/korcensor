import csv

with open('paragraph.csv', mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file, delimiter=',')

    list = []
    for row in csv_reader:
        count = 0
        #print(row)
        srg = row[0].split('.')
        for sen in srg:
            list.append(sen)

with open('sentence.csv', mode='w', encoding='utf-8') as file:
    wr = csv.writer(file)
    for srg in list:
        wr.writerow([srg])
    print(list)


