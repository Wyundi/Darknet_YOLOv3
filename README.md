# Darknet_YOLOv3
Personal modify on YOLOv3_Darknet.



整理darknet框架结构，向其中添加头文件



参考：

- [yolo视频检测之接口实现](https://blog.csdn.net/luoying_ontheroad/article/details/81710646)
- [yolo源码解析(1):代码逻辑](https://blog.csdn.net/JarvisLau/article/details/79939143)



### 添加到example



### 添加到src



- 在 src 目录下新建 my_test.c 和 my_test.h

	- ```c
		// my_test.h
		
		#include "darknet.h"
		```

	- ```c
		// my_test.c
		
		#include <stdio.h>
		
		void my_test_print()
		{
		    printf("test in src.\n");
		}
		```

- 修改 Makefile

	- 在 OBJ 中添加对应 .o 文件

	- ```makefile
		...
		
		OBJ=my_test.o gemm.o ...
		```

- 在 darknet.c 中进行调用

	- ```
		...
		
		else if (0 == strcmp(argv[1], "detect"))
		{
		    float thresh = find_float_arg(argc, argv, "-thresh", .5);
		    char *filename = (argc > 4) ? argv[4]: 0;
		    char *outfile = find_char_arg(argc, argv, "-out", 0);
		    int fullscreen = find_arg(argc, argv, "-fullscreen");
		    my_test_print();
		    test_detector("cfg/coco.data", argv[2], argv[3], filename, thresh, .5, outfile, fullscreen);
		}
		```

- 编译



### 添加到新文件夹