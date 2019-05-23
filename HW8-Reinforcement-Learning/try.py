inputTable=[['Boston', 'Mexican', '163'], ['Boston', 'Seafood', '194'], ['Los Angeles', 'American', '1239'], ['Los Angeles', 'Mexican', '1389'], ['Los Angeles', 'Seafood', '456']]
name=[]
category=[]
for row in inputTable:
    #print(row)
    name_i=row[0]
    cate_i=row[1]
    if(name_i not in name):
        name.append(name_i)
    if(cate_i not in category):
        category.append(cate_i)

name=sorted(name)
category=sorted(category)
list = [ [ 0 for i in range(len(category)) ] for j in range(len(name)) ]

for line in inputTable:
    name_i=line[0]
    cate_i=line[1]
    num=line[2]
    nameindex=name.index(name_i)
    cateindex=category.index(cate_i)
    result=list[nameindex][cateindex]+int(num)
    list[nameindex][cateindex]=result

print(list)