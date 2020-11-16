# Darknet_YOLOv3 框架学习笔记
Personal modify on YOLOv3_Darknet.





###### 参考：

[Darknet 框架学习笔记 ---- 1](https://blog.csdn.net/Jeff_zjf/article/details/101837477)

[Darknet 框架学习笔记 ---- 2](https://blog.csdn.net/Jeff_zjf/article/details/102671661?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-4.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-4.channel_param)

[Darknet 框架学习笔记 ---- 3](https://blog.csdn.net/Jeff_zjf/article/details/102800358?utm_medium=distribute.pc_relevant.none-task-blog-title-1&spm=1001.2101.3001.4242)

[Darknet 框架学习笔记 ---- 4](https://blog.csdn.net/Jeff_zjf/article/details/102887673?utm_medium=distribute.pc_relevant.none-task-blog-title-2&spm=1001.2101.3001.4242)

[训练 Pascal VOC 格式的数据](https://www.cnblogs.com/pprp/p/9525508.html)

[darknet优化经验 (参考YOLOv4)](https://www.cnblogs.com/pprp/p/10204480.html)

[darknet深度学习框架源码分析](https://github.com/hgpvision/darknet)



###### 待查阅：

[【AlexeyAB DarkNet框架解析】一，框架总览](https://cloud.tencent.com/developer/article/1587991)

[【翻译】手把手教你用AlexeyAB版Darknet](https://cloud.tencent.com/developer/article/1575155)



[Pytorch YOLO项目推荐 (U版pytorch_yolov3)](https://cloud.tencent.com/developer/article/1589052)



[【从零开始学习YOLOv3】1. YOLOv3的cfg文件解析与总结](https://cloud.tencent.com/developer/article/1582755)



[目标检测学习路线](https://cloud.tencent.com/developer/article/1548856)





###### 环境：

Ubuntu20.04

OpenCV4.4.0

NVIDIA RTX 2080Ti x2

NVIDIA Driver Version: 450.66

CUDA Version: 10.2



###### 分支：

- master：主分支
- develop：开发分支，包含当前文档
- ori：原版darknet
- serial：原版基础上添加串口输出
- structure：在原版基础上添加子模块方法整理



###### 欠缺：

- cfg文件解读
- anchors计算



###### 注意：

- 搜索 [待整理] 注释，完善相关内容
- 搜索 [待测试] 注释，完成相关测试
- 搜索 [待改进] 注释，完善相关内容





## 建立项目



### 创建仓库

下载源码，重新建立项目结构并上传到github

```bash
# 下载源码
git clone https://github.com/pjreddie/darknet

# 初始化项目
cd darknet
rm -rf .git
git init

# 配置 .gitignore, 添加 models/, weights/
vim .gitignore
git add .

# 提交到git仓库
git commit -m "first commit"

# 关联到github
git remote add origin https://github.com/Wyundi/Darknet_YOLOv3.git
git pull --rebase origin master
git push origin master

# 建立新分支
# develop分支用于开发，master分支用于新功能发布，ori分支用于保存原有darknet框架
git checkout -b develop
git checkout -b ori

# 删除分支
git branch -d [branch name]
```



### 配置Makefile



#### 修改Makefile

```makefile
# Makefile 修改如下：
# 打开GPU，CUDNN，OPENCV
GPU=1
CUDNN=1
OPENCV=1
...

# 配置显卡算力
ARCH= -gencode arch=compute_75,code=compute_75
...

# 更改opencv版本
LDFLAGS+= `pkg-config --libs opencv4` -lstdc++
COMMON+= `pkg-config --cflags opencv4`
```



#### 解析Makefile



GPU，CUDNN，OPENCV，OPENMP，DEBUG，ARCH，SLIB，ALIB，EXEC，OBJDIR，OPTS，COMMON等都是自动变量

ARCH：

- 在ARCH中-gencode保证用户GPU可以动态选择最适合的GPU框架，`-arch=compute_30，-code=sm_30`表示计算能力3.0及以上的GPU都可以运行编译的程序。
- NVIDIA GPU算力可由此查看：[查看显卡算力](https://developer.nvidia.com/cuda-gpus)
- GTX2080Ti 算力7.5，于是将其改为

```
ARCH= -gencode arch=compute_75,code=compute_75
```



> 
>
> SLIB指明动态链接库文件，ALIB指明静态链接库文件，EXEC指明可行文件名称，OBJDIR指明中间文件存放目录
>
> OPTS指明编译选项，OPTS=-Ofast中-Ofast 表示忽视严格的标准，使用所有-O3优化。-[Ofast详见此](http://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)
>
> 
>
> VPATH，CC，CCP，NVCC，AR，LDFLAGS，CFLAGS等是内置变量
>
> VPATH指定依赖文件的搜索路径，当有多个路径时用分号 : 隔开
>
> AR命令可以用来创建、修改库，也可以从库中提出单个模块，参考[此博客](https://www.cnblogs.com/LiuYanYGZ/p/5535982.html)
>
> ARFLAGS是库文件维护程序的选项
>
> LDFLAGS是GCC链接选项参数，LDFLAGS= -lm -pthread，-lm代表链接数学函数库，-pthread代表链接多线程编译库
>
> CFLAGS是C编译选项
>
> addprefix函数是添加前缀
>
> wildcard是扩展通配符，$(wildcard src/*.h)代表src/下的头文件
>
> 这是Makefile的执行主题，分号前面是“目标”，分号后面是“依赖”，下一行是“命令”
>
> 
>
> Makefile有三个非常有用的变量。分别是$@，$^，$<代表的意义分别是：
>
> $@--目标文件，$^--所有的依赖文件，$<--第一个依赖文件。
>
> $(OBJDIR)%.o 代表文件夹下所有的.o文件
>
> mkdir的-p选项允许你一次性创建多层次的目录
>
> 
>
> 当Makefile文件所在目录有文件名为clean的文件，命令行“.PHONY: clean”又没添加的话，执行make clean是无效的
>
> 所以“.PHONY: clean”就是保证即使目录下有文件名为clean的文件，也能正常执行make clean
>
> -DGPU相当于添加了GPU宏定义，用来条件编译和预处理。



### 编译文件

- 使用make命令编译文件

- 编译报错：

  - warning: ‘cudaThreadSynchronize’ is deprecated

    > Just replace at /src/gemm.c:232: cudaThreadSynchronize() with cudaDeviceSynchronize().

  - error: ‘IplImage’ does not name a type

    - 因OpenCV版本问题，部分参数和文件需要修改

    - 添加头文件：\#include "opencv2/imgproc/imgproc_c.h"

    - 函数 open_video_stream 中，参数 CV_CAP_PROP_FRAME_WIDTH / CV_CAP_PROP_FRAME_HEIGHT / CV_CAP_PROP_FPS, 以及函数 make_window 中，参数 CV_WND_PROP_FULLSCREEN / CV_WINDOW_FULLSCREEN 需要将 [CV_] 前缀删除。具体修改如下：

      ```c
      ...
      # 添加头文件
      #include "opencv2/imgproc/imgproc_c.h"
      
      ...
      # 删除参数中的 [CV_]
      void *open_video_stream(const char *f, int c, int w, int h, int fps)
      {
          VideoCapture *cap;
          if(f) cap = new VideoCapture(f);
          else cap = new VideoCapture(c);
          if(!cap->isOpened()) return 0;
          if(w) cap->set(CAP_PROP_FRAME_WIDTH, w);
          if(h) cap->set(CAP_PROP_FRAME_HEIGHT, w);
          if(fps) cap->set(CAP_PROP_FPS, w);
          return (void *) cap;
      }
      
      ...
      # 删除参数中的 [CV_]
      void make_window(char *name, int w, int h, int fullscreen)
      {
          namedWindow(name, WINDOW_NORMAL); 
          if (fullscreen) {
              setWindowProperty(name, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
          } else {
              resizeWindow(name, w, h);
              if(strcmp(name, "Demo") == 0) moveWindow(name, 0, 0);
          }
      }
      ```

- 重新使用make命令编译文件，编译通过



## 使用darknet进行训练和预测



### 训练



#### 下载预训练模型并提取权重

注意：官方提供的模型对应的数据集为coco数据集，测试应使用cfg/coco.data

```bash
# 下载预训练模型
wget https://pjreddie.com/media/files/yolov3.weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights

# 提取卷积层权重
./darknet partial cfg/yolov3.cfg weights/yolov3.weights weights/darknet53.conv.74 74
./darknet partial cfg/yolov3-tiny.cfg weights/yolov3-tiny.weights weights/yolov3-tiny.conv.15 15
```



#### 整理数据

- VOC格式数据不能直接用于darknet训练，需要使用 [voc_label.py](####voc_label.py) 将其转换为darknet格式
- 计算anchors并写入cfg文件



#### 创建配置文件

- voc.names 文件中存放待训练的样本类别

- voc.data 文件中存放基本训练信息，包含以下信息：

  - classes: 类别数
  - train: 训练集文件地址
  - valid: 验证集文件地址
  - names: 类别名称文件地址
  - backup: 训练权重保存地址

- yolov3-voc.cfg 文件中存放网络结构配置文件，需要修改其中的anchors, classes, filters, 以及batch_size， 训练步数等。注意每一个 [yolo] 节点之前的 filters 要改为如下数据：

  - filters = 3 * ( classes + 5 )

    ```bash
    [net]
    # Testing
    # batch=64
    # subdivisions=32
    # Training
    batch=64
    subdivisions=16					#每批训练的个数=batch/subvisions，根据自己GPU显存进行修改，显存不够改大一些
    width=416
    height=416
    channels=3
    momentum=0.9
    decay=0.0005
    angle=0
    saturation = 1.5
    exposure = 1.5
    hue=.1
    
    learning_rate=0.001
    burn_in=1000
    max_batches = 50200  #训练步数
    policy=steps
    steps=40000,45000  #开始衰减的步数
    scales=.1,.1
    
    ...
    
    [convolutional]
    size=1
    stride=1
    pad=1
    filters=24   #filters = 3 * ( classes + 5 )   here,filters=3*(3+5)
    activation=linear
    
    [yolo]
    mask = 6,7,8
    anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
    classes=3    #修改为自己的类别数
    num=9
    jitter=.3
    ignore_thresh = .5
    truth_thresh = 1
    random=1
    
    ...
    ```

    

#### 训练（多GPU）

- 生成的模型存放于 voc.data 中 backup 定义的位置
- -gpus 参数用于指定训练需要调用的显卡

```bash
# norm
./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg weights/darknet53.conv.74 -gpus 0,1

# tiny
./darknet detector train cfg/voc.data cfg/yolov3-tiny.cfg weights/yolov3-tiny.conv.15 -gpus 0,1
```

- 训练中断后从中断处继续训练

```bash
./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg models/model.backup -gpus 0,1
```



### 预测

仅用常规模型举例，tiny版本换成对应的文件即可



- 单张图片进行预测

  - ```bash
    ./darknet detector test cfg/coco.data cfg/yolov3.cfg weights/yolov3.weights image.jpg
    ```

- 视频流预测

  - ```bash
    # 视频文件
    ./darknet detector demo cfg/coco.data cfg/yolov3.cfg weights/yolov3.weights ~/Videos/video.mp4
    
    # usb相机
    # 0 表示 "/dev/video0", 也可直接使用 "/dev/video0"
    ./darknet detector demo cfg/coco.data cfg/yolov3.cfg weights/yolov3.weights -c 0
    
    # webcamera
    # 海康："rtsp://admin:tiyoa000@192.168.0.64/Streaming/Channels/1"
    # 天地伟业："rtsp://Admin:1111@192.168.0.2:554/1/1"
    ./darknet detector demo cfg/coco.data cfg/yolov3.cfg weights/yolov3.weights "rtsp://admin:tiyoa000@192.168.0.64/Streaming/Channels/1"
    ```

- 添加功能模块：文件夹预测

  - ```
    ./darknet detector test_file cfg/coco.data cfg/yolov3.cfg weights/yolov3.weights filename/
    ```

- 其余功能 [待整理]



#### 参数



##### 一级参数：供 ./darknet 调用

- -i: 声明GPU索引，供预测过程使用
- -i 0: 指定使用第一块GPU
  - -i 1: 指定使用第二块GPU
  - 不添加 -i 参数：默认调用第一块GPU
  - 多CPU并行运算仅支持训练过程，预测过程无法调用多块GPU。查看参数 [-gpus]
- -nogpu: 不调用GPU



##### 二级参数：供 detector 调用

- -prefix: 控制视频流输出方式

  - 仅对视频流检测有效 [detector demo]

  - 使用方法：调用命令 [ detector demo ... -prefix ~/Video/test/prefix ] 对视频流进行预测

  - 作用：如果参数包含 [prefix] ，则将每一帧的处理结果保存成文件，不在屏幕进行展示。文件前缀为 [-prefix] 后跟的参数，格式为 [.jpg]

  - 范例：

    - ```bash
      $ ./darknet detector demo cfg/coco.data cfg/yolov3.cfg weights/yolov3.weights ~/Videos/video.mp4 -prefix ~/Video/test/prefix
      ```

    - 调用上面的命令，则检测结果会保存为 [~/Video/test/prefix_00000000.jpg, ~/Video/test/prefix_00000001.jpg, ...] ，直到视频结束

- -thresh: 设置输出置信度

  - 默认值为 [0.5]
  - 检测置信度小于 [-thresh] 的将不被输出

- -hier: 涉及到目标检测结果中父类和子类的概念。具体查看 [[源码解析](####run_detector())] 中相关内容。

- -c: 声明相机索引

  - -c 0
  - -c "/dev/video0"

- -s: frame_skip, 跳过帧。可以设定这个参数，但在源码中没有调用，因此这个参数目前无效 [待改进]

- -avg: 后续代码中没有调用。默认值为3

- -gpus: 声明GPU索引，仅供训练过程配置GPU并行运算

- -out: 声明测试完成后结果保存位置。供单张图片测试 [test] 和验证 [valid / valid2] 使用。

  - [test] 中：默认保存位置为根目录下 [./predictions]
  - [valid / valid2]中：默认保存位置为从cfg文件中获取的prefix

- -clear: 如果训练意外中断，并使用 [yolov3.backup] 文件继续训练，加上 [-clear] 参数可以初始化迭代次数和学习率。仅供训练过程使用。[待测试]

- -fullscreen: 设置是否全屏显示

- -w, -h: 设置相机分辨率

  - -w 1920 -h 1080

- -fps: 设置输入视频流帧速率

  



#### 输出

- 使用bash管道将预测结果输出到文件

  - ```bash
    ./darknet detector test cfg/coco.data cfg/yolov3.cfg weights/yolov3.weights image.jpg > ~/Desktop/res.txt
    ```

- 添加功能模块：输出到串口

  - [待整理]



## 源码解析



### 工程文件结构

- cfg/: 目录下存放.cfg文件，包含网络结构，学习参数等
- data/: 存放训练数据和标注文件，.names 文件存放类别名。labels/ 目录下有ASCII码32-127的8种尺寸的图片，显示标签用
- examples/: 例程代码
- include/: darknet.h
- obj/: 编译中间文件
- python/: darknet.py, proverbot.py
- scripts/: 有关训练集索引文件的脚本
- src:/: 源码
- backup/: 初始为空，用于保存训练产生的权重文件



**添加目录：**

- weights/: 保存初始网络模型和卷积部份权重
- models/: 保存不同项目模型及其配置文件





### 主函数：main()

##### 位置：examples/derknet.c

##### 参数：int argc, char **argv

##### 功能：作为程序入口，根据命令行参数调用子函数

- 检查参数列表
- 配置GPU索引
- 调用子函数

##### 子函数：detector()

- [detector](####run_detector()): 训练，验证和测试
  - 训练：train()
  - 验证：valid() / valid2() / recall()
  - 测试：test() / demo()

##### 执行顺序：检查参数列表，配置GPU，将参数传递给子函数

- 检查参数列表

  - 参数少于2个：输出错误信息并返回

- 配置GPU

  - 查找参数 [-i] / [-nogpu]，并根据其结果设定[GPU索引 [gpu_index]](####gpu_index) 
    - -i: 指定GPU索引：将GPU索引设定为 -i 后面的参数值
    - -nogpu: 不调用GPU：将GPU索引设定为 -1
    - 不包含这两个参数：将GPU索引设定为 0
    - 注：查找参数需要调用函数 [find_arg()](#####find_arg():), [find_int_arg()](#####find_int_arg() / find_float_arg() / find_char_arg():)
  - if(gpu_index >= 0):
    - 调用CUDA配置GPU：cuda_set_device(gpu_index);

- 将参数传递给子函数

  - detector: 调用 [run_detector()](####run_detector())
    - 传入参数：argc, argv

- 获得错误参数

  - > fprintf(stderr, "Not an option: %s\n", argv[1]);

- return 0;



### 应用层



#### run_detector()



##### 位置：examples/detector.c

##### 参数：int argc, char **argv

##### 功能：训练 / 验证模型并在图片和视频流上进行测试

- 检查参数列表
- 配置GPU索引
- 调用子函数

##### 子函数：训练 / 验证 / 测试

- [test](####test_detector()): 在单张图片上测试识别效果
- [train](####train_detector()): 训练网络模型
- [valid](####validate_detector()): 验证模型
- [valid2](####validate_detector_flip): 验证模型
- [recall](####validate_detector_recall()): 验证模型
- [demo](####demo()): 在视频流上测试识别效果

##### 执行顺序：查找参数，配置GPU索引，将参数传递给子函数

- 查找参数:

  - prefix: 共视频流检测 [demo] 调用，如果参数包含 [prefix] ，则将处理结果输出到文件，不在屏幕进行展示。具体使用细节查看[参数](####参数)。
  - thresh: [float] 类型参数，设置检测阈值。置信度小于 [thresh] 的将不被输出。
  - hier_thresh: hierarchy_top_prediction, [float] 类型参数，涉及到目标检测结果中父类和子类的概念。如果检测置信度高于 [hier] 的设定值，输出子类的标签，否则输出父类标签。例如：设置 [hier] 参数为 [0.9] ，检测到样本类别为 [people] ，置信度为 [1.0] 。同时，检测到样本类别为 [people] 的子类 [woman] ，若 [woman] 的置信度大于设定值 [0.9] ，则输出标签 [woman] ，否则输出标签 [people] 。[待测试]
  - cam_index: 声明相机索引
  - frame_skip: 跳过帧，但在后续代码中没有调用。
  - avg: 后续代码中没有调用。默认值为3

- 检查参数列表

  - 参数少于4个：输出错误信息并返回

- 查找参数：

  - gpu_list: 声明GPU索引，仅供训练过程配置GPU并行运算
  - outfile: 供单张图片测试 [test] 和验证 [valid / valid2] 使用。声明测试完成后结果保存位置。具体使用细节查看[参数](####参数)。

- 配置GPU索引:

  - 如果参数包含 [gpu_list] ：

    - 将 [gpu_list] 中的数值存入 [gpus] 中
    - GPU数量存入 [ngpus] 中

  - 如果不包含：

    - ```c
      gpu = gpu_index;
      gpus = &gpu;
      ngpus = 1;
      ```

- 查找参数：

  - clear: 供训练过程使用，用于初始化backup文件中保存的迭代次数和学习率。具体使用细节查看[参数](####参数)。
  - fullscreen: 设置是否全屏显示
  - width / height: 设置输入视频流分辨率
  - fps: 设置输入视频流帧速率

- 读取参数列表：

  - ```c
    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    ```

- 将参数传递给子函数

  - test: 调用 [test_detector()](####test_detector())
    - 传入参数：datacfg, cfg, weights, filename, thresh, hier_thresh, outfile, fullscreen
  - train: 调用 [train_detector()](####train_detector())
    - 传入参数：datacfg, cfg, weights, gpus, ngpus, clear
  - valid: 调用 [validate_detector()](####validate_detector())
    - 传入参数：datacfg, cfg, weights, outfile
  - valid2: 调用 [validate_detector_flip()](####validate_detector_flip())
    - 传入参数：datacfg, cfg, weights, outfile
  - recall: 调用 [validate_detector_recall()](####validate_detector_recall())
    - 传入参数：cfg, weights
  - demo: 
    - 调用 [read_data_cfg()](####read_data_cfg()) 读取 [datacfg] ，存入 [options]
    - 调用 [option_find_int()](####option_find_str / option_find_int() / option_find_float()) 读取类别数量，默认值为20，存入 [classes]
    - 调用 [option_find_str()](####option_find_str / option_find_int() / option_find_float()) 读取类别名称 ，存入 [name_list]。默认为 [data/names.list]
    - 调用 [get_labels()](####get_labels()) 将 [name_list] 转化为 [names]
    - 调用 [demo()](####demo())
      - 传入参数：cfg, weights, thresh, cam_index, filename, names, classes, frame_skip, prefix, avg, hier_thresh, width, height, fps, fullscreen



#### test_detector()

##### 位置：examples/detector.c

##### 参数：char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen

##### 功能：对图片文件进行识别

##### 执行顺序：加载图片，加载网络结构，识别并输出

- 调用 [read_data_cfg()](####read_data_cfg()) 读取 [datacfg] ，存入 [options]
- 调用 [option_find_str()](####option_find_str / option_find_int() / option_find_float()) 读取类别名称 ，存入 [name_list]。默认为 [data/names.list]
- 调用 [get_labels()](####get_labels()) 将 [name_list] 转化为 [names]
- 调用 [load_alphabet()](#####load_alphabet()) 将字符对应的图片存入 [image] 类型的结构体指针 [alphabet] 中
- 调用 [load_network()](####load_network()) 加载网络结构，存入 [network] 类型的结构体指针 [net] 中
- 调用 [set_batch_network()](####set_batch_network()) 设置网络结构中的 [batch]
- 调用 [srand()](#####srand()) 生成随机数
- 声明 [time]，[input]，[nms]
  - [time] 用于记录系统时间
  - [input] 用于存储输入的文件名
  - [nms] 非极大值抑制
- 进入循环：
  - 读取（或请求输入）文件名
  - 调用 [load_image_color()](#####load_image_color()) 加载图片，存入 [image] 类型的结构体 [im] 中
  - 调用 [letterbox_image()](#####letterbox_image()) 将图片大小调整为网络结构中要求的大小，并将调整好的图片存入 [image] 类型的结构体 [sized] 中
  - 从 [net] 中获取层结构
  - 从 [sized] 中读取数据，存入 [X]
  - [开始计时] -> 将数据传入网络进行预测 -> [结束计时]
    - 记录时间：[what_time_is_it_now()](#####what_time_is_it_now())
    - 将 [X] 输入到函数 [network_predict()](####network_predict()) 中进行预测
    - 输出预测所需时间
  - 调用 [get_network_boxes()](####get_network_boxes()) 从 [net] 中获取预测结果，存入 [detection] 类型的结构体指针 [dets] 中 [待整理]
  - 调用 [do_nms_sort()](####do_nms_sort()) 对结果进行非极大值抑制处理
  - 调用 [draw_detections()](#####draw_detections()) 对结果进行绘制
  - 调用 [free_detections()](#####free_detections()) 释放 [dets] 指针
  - 保存图片：调用 [save_image()](#####save_image()) ，如果声明了 [outfile] 则将图片保存到声明的位置，否则保存到默认位置
  - 如果声明了 [OpenCV]：
    - 调用 [make_window()](#####make_window()) 创建窗口
    - 调用 [show_image()](#####show_image()) 显示图片
  - 调用 [free_image()](#####free_image()) 释放 [im], [sized]
  - 调用 [ if(filename) ] 确保文件被正确读取，然后跳出循环



#### train_detector()





#### validate_detector()





#### validate_detector_flip()





#### validate_detector_recall()





#### demo()

##### 位置：src/demo.c

##### 参数：char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen

##### 功能：对视频文件/视频流进行实时识别

##### 执行顺序：

- 调用 [load_alphabet()](#####load_alphabet()) 将字符对应的图片存入 [image] 类型的结构体指针 [alphabet] 中
- 初始化 [demo_names]
- 初始化 [demo_alphabet]
- 初始化 [demo_classes]
- 初始化 [demo_thresh]
- 初始化 [demo_hier]
- 调用 [load_network()](####load_network()) 加载网络结构，存入 [network] 类型的结构体指针 [net] 中
- 调用 [set_batch_network()](####set_batch_network()) 设置网络结构中的 [batch]
- 声明线程 [detect_thread]
- 声明线程 [fetch_thread]
- 调用 [srand()](#####srand()) 生成随机数
- 根据网络结构分配内存空间，供 [predictions] 使用
- 调用 [open_video_stream()](#####open_video_stream()) 读取视频流，视频流格式可以是视频文件，相机或者网络视频流等
- 调用 [get_image_from_stream()] 从视频流中获取图像
- 调用 [copy_image] 将图片复制到其余两个线程
- 调用 [letterbox_image] 整理三个线程中图片的格式
- if(!prefix): 创建视频流显示窗口
- 调用 [what_time_is_it_now()](#####what_time_is_it_now()) 获取当前时间
- 进入检测循环：
  - 获取当前线程索引，存于 [buff_index] 中
  - 调用 [pthread_create()](#####pthread_create()) 创建线程 [fetch_thread]，执行子函数 [fetch_in_thread()](#####fetch_in_thread())，从视频流中读取图像
  - 调用 [pthread_create()](#####pthread_create()) 创建线程 [detect_thread]，执行子函数 [detect_in_thread()](#####detect_in_thread())，对图像进行识别
  - 计算 [fps]
  - 调用 [what_time_is_it_now()](#####what_time_is_it_now()) 初始化当前时间
  - if(!prefix): 执行子函数 [display_in_thread()](#####display_in_thread())，else: 保存当前帧到指定目录
  - 调用 [pthread_join()](#####pthread_join) 等待子线程结束





### 功能函数



#### 时间记录：

##### 位置：src/utils.c



##### what_time_is_it_now():

参数：无

功能：

- 获取当前时间



#### 参数相关函数：

##### 位置：src/utils.c



##### del_arg():

参数：int argc, char **argv, int index

功能：

- 将 index 位置的参数删除



##### find_arg():

参数：int argc, char* argv[], char *arg

功能：

- 查找字符串 arg 是否存在于参数列表中：
  - 存在：调用 [del_arg()](#####del_arg()) 将其删除，并 return 1
  - 不存在：return 0



##### find_int_arg() / find_float_arg() / find_char_arg():

参数：int argc, char **argv, char *arg, (int def / float def / char *def)

功能：

- 查找字符串 arg 是否存在于参数列表中：
  - 存在：将 def 的值变更为参数 arg 后面一个参数包含的值，并调用 [del_arg()](#####del_arg()) 将这两个参数从参数列表中删除
  - 不存在：def 保持为传入的默认值
- return def



#### 配置文件相关函数：

##### 位置：src/option_list.c



##### read_data_cfg():

参数：char *filename

功能：

- 读入cfg配置文件，存入链表 [list]



##### option_find():

参数：list *l, char *key

功能：

- 从 [list] 中查找 [key]
  - [list] 中包含 [key] ：返回 [key] 对应的 [val]
  - 不包含：返回 0



##### option_find_str() / option_find_int() / option_find_float():

参数：list *l, char *key, (char *def, int def, float def)

功能：

- 从 [option] 中查找 [key]：
  - 如果包含 [key] ：将结果转变为对应的类型返回
  - 如果不包含：向 [stderr] 中输出 [ "%s: Using default %d", key, def ]
  - return def



##### option_find_int_quiet() / option_find_float_quiet():

参数：list *l, char *key, (char *def, int def, float def)

功能：

- 从 [option] 中查找 [key]：
  - 如果包含 [key] ：将结果转变为对应的类型返回
  - 如果不包含：return def



#### 数据相关函数：

##### 位置：src/data.c



##### get_labels():

参数：char *filename

功能：

- 调用 [list_to_array] 函数将存储在 [list] 中的类别信息保存在 [char **labels] 中，并将其返回



#### 图像相关函数：

##### 位置：src/image.c



##### load_alphabet():

参数：无

功能：从目录 [data/labels/] 中加载标准字符对应的不同大小的图像，用于叠加在输出结果中。具体查看目录 [data/labels/]



##### load_image_color():

参数：char *filename, int w, int h

功能：

- 调用 [load_image()](#####load_image()) 加载图像



##### load_image():

参数：char *filename, int w, int h, int c

功能：

- 如果声明了 [OpenCV] 则使用 [OpenCV] 加载图像，否则使用常规方法加载图像
- 如果 [w] 和 [h] 都不为零，且与图像的宽高不相等，则将图像的大小缩放到 (w, h)



##### letterbox_image():

参数：image im, int w, int h

功能：

- 参数 [w] 和 [h] 从 [net->w] 和 [net->h] 中获取
- 将图片按比例缩放到小于等于 [w] 和 [h]
- 创建空白图像 [boxed] ，数据用 [0] 填充
- 将缩放后的图像填入 [boxed] 中，并作为返回值返回





##### open_video_stream():

参数：const char *f, int c, int w, int h, int fps

功能：创建视频流



#### 线程相关函数：

##### pthread_create():





##### fetch_in_thread():

位置：src/demo.c

参数：void *ptr

功能：从视频流中获取图像



##### detect_in_thread():

位置：src/demo.c

参数：void *ptr

功能：对图像进行检测，并输出检测结果



##### display_in_thread():





##### pthread_join():





#### C语言内置函数：

##### strcmp(str1, str2): 

- 检查 [str1]和 [str2] 是否相同



##### atoi() / atof():

- 将 [str] 类型参数转换为 [int] / [float] 类型



##### srand():

- 随机数发生器





### 神经网络



#### load_network():

位置：src/network.c

参数：char *cfg, char *weights, int clear

功能：

- 从 [cfg] 文件中加载网络结构



#### set_batch_network():

位置：src/network.c

参数：network *net, int b

功能：

执行顺序：

- 将 [net->batch] 设置为 [b]
- 将网络中每一层的 [batch] 设置为 [b]
- 如果使用 [CUDNN] ：配置 [CUDNN] [待整理]



#### network_predict():

位置：src/network.c

参数：network *net, float *input

功能：将 [input] 输入神经网络进行预测

执行顺序：

- 在 [orig] 中备份 [net]
- 将 [input] 写入 [net]
- 调用 [forward_network()](####forward_network()) 进行预测
- 将结果保存在 [out] 中
- 用 [orig] 还原 [net]
- return out



#### forward_network():

位置：src/network.c

参数：network *netp

功能：沿网络结构进行前向传播

执行顺序：

- 如果包含 [GPU] ：调用 [forward_network_gpu()](####forward_network_gpu()) 在GPU中进行运算
- 如果不包含：
  - 在CPU中进行运算 [待整理]



#### get_network_boxes():

位置：src/network.c

参数：network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num

功能：根据网络输出，提取检测到的目标位置以及类别信息

执行顺序：

- 调用 [make_network_boxes()](####make_network_boxes()) 创建 [detection] 类型的结构体指针 [dets]
- 调用 [fill_network_boxes()](####fill_network_boxes()) 从网络中获取预测结果
- return dets



#### make_network_boxes():

位置：src/network.c

参数：network *net, float thresh, int *num

功能：

- [detection] 类型的结构体指针 [dets] 并分配地址空间



#### fill_network_boxes():

位置：src/network.c

参数：network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets

功能：从网络中获取预测结果 [待整理]

执行顺序：

-  对网络中每一层：
  - if [l.type == YOLO] -> 调用 [get_yolo_detections()](####get_yolo_detections()) ，并将结果添加到 [dets] 
  - if [l.type == REGION] -> 调用 [get_region_detections()](####get_region_detections()) ，并将结果添加到 [dets] 
  - if [l.type == DETECTION] -> 调用 [get_detection_detections()](####get_detection_detections()) ，并将结果添加到 [dets] 



#### get_yolo_detections():

位置：src/yolo_layer.c

参数：layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets

功能：

执行顺序：





#### get_region_detections():

位置：src/region_layer.c

参数：layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets

功能：

执行顺序：





#### get_detection_detections():

位置：src/detection_layer.c

参数：layer l, int w, int h, float thresh, detection *dets

功能：

执行顺序：







### 结构体



#### image：

```c
typedef struct {
    int w;
    int h;
    int c;
    float *data;
} image;

```



#### network：

```c
typedef struct network{
    int n;
    int batch;
    size_t *seen;
    int *t;
    float epoch;
    int subdivisions;
    layer *layers;
    float *output;
    learning_rate_policy policy;

    float learning_rate;
    float momentum;
    float decay;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    float *scales;
    int   *steps;
    int num_steps;
    int burn_in;

    int adam;
    float B1;
    float B2;
    float eps;

    int inputs;
    int outputs;
    int truths;
    int notruth;
    int h, w, c;
    int max_crop;
    int min_crop;
    float max_ratio;
    float min_ratio;
    int center;
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;
    int random;

    int gpu_index;
    tree *hierarchy;

    float *input;
    float *truth;
    float *delta;
    float *workspace;
    int train;
    int index;
    float *cost;
    float clip;

#ifdef GPU
    float *input_gpu;
    float *truth_gpu;
    float *delta_gpu;
    float *output_gpu;
#endif

} network;
```





#### detection：

```c
typedef struct detection{
    box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
} detection;
```





### 数据预处理工具



#### voc_label.py：

- 位置：scripts
- 功能：将VOC格式数据转换为darknet所需的格式