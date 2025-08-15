### coco_to_yolo_quadrant，coco_to_yolo_quadrant-enumeration，coco_to_yolo_quadrant-enumeration-disease这三个是dentalX的，放在training_data文件夹下对应的文件夹里。运行后在对应文件夹里创建yolo文件夹，并把json文件转换为yolo格式的标签，存储在该文件夹里，同时会生成classes.txt和dataset.yaml文件，但此时还不是yolo格式要求的文件夹结构。需要在yolo文件夹下再创建images文件夹，并且在labels和images文件夹下分别创建train和val文件夹，自行把想要用来训练和验证的图片和标签放到对应的文件夹里，图片和标签要一一对应。

### coco2yolo_train，coco2yolo_val这两个是OrayXrays9的，下载OrayXrays的annotations文件并解压后，把这两个文件放在annotations文件夹下，运行后会创建yolo文件夹并把json文件转换为yolo格式，存储在该文件夹下（另外，运行时可能会提示“警告: 未知的category_id ，已跳过该标注”，是因为原json文件里就有一个没有指明的种类，我没找到它是干什么的，所以就跳过了）。然后把yolo文件夹下的labels_train2017重命名为train，labels_val2017重命名为val。再把文件夹结构调整为：

yolo/  
├── images/  
│   ├── train/                 # 训练图片（10000张）  
│   └── val/                   # 验证图片（2688张）  
├── labels/                       
│   ├── train/                 # 上面提到的重命名后的train文件夹（10000个标签）  
│   └── val/                   # 上面提到的重命名后的val文件夹（2688个标签）  
└── dataset.yaml               # 在运行coco2yolo_train时会生成这个文件，把它放在这个位置就可以  

做完这些之后，利用  
python scripts/train.py --data_dir "G:\pycharm\项目\dental\yolov8_teeth\OralXrays9\dataset.yaml"  
命令（引号里那一串路径需要替换为.yaml文件的路径）就可以开始训练了（或许在这之前还有虚拟环境，某些参数的调整之类的，但是那些我就不知道你们应该怎么搞了）