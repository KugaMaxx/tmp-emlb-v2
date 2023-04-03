如果想编译带有神经网络的算法，那么请使用如下代码
```bash
mkdir build
cd build
cmake .. -DTORCH_DIR:STRING=/usr/local/cuda/include
```