original_list, remove_list = [], []
original_txt_file = open("annotation.txt", 'r')
remove_txt_file = open("annotation_letters.txt", 'r')

while True:
    line = original_txt_file.readline()
    split_line = line.strip().split(',')
    if not line:
        break
    original_list.append(split_line)

while True:
    line = remove_txt_file.readline()
    split_line = line.strip().split(',')
    if not line:
        break
    remove_list.append(split_line)

mixed_list = []
for i in range(len(original_list)):
    for j in range(len(remove_list)):
        if original_list[i][0] == remove_list[j][0]:
            if original_list[i][1] != '0':
                mixed = original_list[i]+remove_list[j][2:]
                mixed[1] = str(int(mixed[1])+1)
                mixed_list.append(mixed)
                break
        else:
            if j == len(remove_list)-1:
              mixed_list.append(original_list[i])  

mixed_list_to_csv = []
for i in range(len(mixed_list)):
    tmp = ""
    for j in range(len(mixed_list[i])):
        if j == len(mixed_list[i])-1:
            tmp += mixed_list[i][j]
        else:
            tmp += (mixed_list[i][j]+",")
    mixed_list_to_csv.append(tmp)

with open('annotation_label_letters.txt','w',encoding='UTF-8') as f:
    for line in mixed_list_to_csv:
        f.write(line+'\n')

original_txt_file.close()
remove_txt_file.close()