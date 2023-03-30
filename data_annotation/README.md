# Data Annotator & .txt concatenator

## 1. Data Annotator
```python
print("1. 입력한 파라미터인 이미지 경로(--path)에서 이미지들을 차례대로 읽어옵니다.")
print("2. 키보드에서 'n'을 누르면(next 약자) 다음 이미지로 넘어갑니다. 이 때, 작업한 점의 좌표가 저장 됩니다.")
print("3. 키보드에서 'b'를 누르면(back 약자) 직전에 입력한 좌표를 취소한다.")
print("4. 이미지 경로에 존재하는 모든 이미지에 작업을 마친 경우 또는 'q'를 누르면(quit 약자) 프로그램이 종료됩니다.")
print("출력 포맷 : 이미지명,점의갯수,y1,x1,y2,x2,...")
```
## 2. .txt concatenator
```python
original_txt_file = open("path/annotation_file_name.txt", 'r')
remove_txt_file = open("path/remove_annotation_file_name.txt", 'r')
...
with open('path/destination_file_name.txt','w',encoding='UTF-8') as f:
    for line in mixed_list_to_csv:
        f.write(line+'\n')
```
