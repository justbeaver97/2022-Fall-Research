```
data
├─ annotation_text_files
|   └─ ...
├─ dataset 
│   ├─ annotation
|   |   └─ ...
│   ├─ image
|   |   └─ ...
├─ dicom_data
|   └─ ...
├─ dicom_to_png
│   ├─ annotation_image
|   |   └─ ...
│   ├─ original_image
|   |   └─ ...
├─ overlay_image_to_label
|   └─ ...
├─ overlay_only
|   └─ ...
└─ padded_image
    └─ ...
```

- `annotation_text_files`: Pixel value of all the labels in the image
```
0_pad.png,6,91,1196,1476,1005,1463,1167,1611,922,1599,1165,2435,1182
100_pad.png,0
101_pad.png,6,28,987,919,830,929,1027,1011,731,1028,1012,1959,993
103_pad.png,6,34,1213,1135,1007,1143,1181,1228,917,1243,1174,2256,1119
104_pad.png,6,128,1312,1398,1120,1384,1319,1513,997,1489,1318,2542,1328
106_pad.png,6,36,913,904,802,910,978,1020,710,1023,978,1853,1002
107_pad.png,6,195,867,1198,624,1184,857,1304,510,1305,859,1718,872
108_pad.png,6,77,1341,1385,1188,1402,1333,1526,1137,1542,1339,2598,1364
109_pad.png,6,233,1248,1457,1067,1438,1249,1594,1035,1579,1249,2419,1331
```
- `dataset`: explanation
- `dicom_data`: dicom data from the medical center
- `dicom_to_png - annotation_image`: change of annotation dicom data into png format
- `dicom_to_png - original_image`: change of original xray dicom data into png format
- `overlay_image_to_label`: overlay annotation and original image from `dicom_to_png`
- `overlay_only`: only the overlay images from `overlay_image_to_label`
- `padded_image`: padding image to make the height and the width be equal to each other