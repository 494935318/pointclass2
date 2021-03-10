#/bin/bash

TF_CFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
CUDA_ROOT=/usr/local/cuda-10.1
echo "Compiling GPU Ops..."

g++ -std=c++11 -shared ./3d_interpolation/tf_interpolate.cpp -o ./3d_interpolation/tf_interpolate_so.so  -I $CUDA_ROOT/include -lcudart -L $CUDA_ROOT/lib64/ -fPIC  -I/home/guo/anaconda3/envs/tensorflow_gpuenv/lib/python3.8/site-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=0 -L/home/guo/anaconda3/envs/tensorflow_gpuenv/lib/python3.8/site-packages/tensorflow -l:libtensorflow_framework.so.2
echo "Interpolate op compiled."

$CUDA_ROOT/bin/nvcc ./grouping/tf_grouping_g.cu -o ./grouping/tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 -shared ./grouping/tf_grouping.cpp ./grouping/tf_grouping_g.cu.o -o ./grouping/tf_grouping_so.so -I $CUDA_ROOT/include -L $CUDA_ROOT/lib64/ -fPIC   -I/home/guo/anaconda3/envs/tensorflow_gpuenv/lib/python3.8/site-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=0 -L/home/guo/anaconda3/envs/tensorflow_gpuenv/lib/python3.8/site-packages/tensorflow -l:libtensorflow_framework.so.2

$CUDA_ROOT/bin/nvcc ./sampling/tf_sampling_g.cu -o ./sampling/tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 -shared ./sampling/tf_sampling.cpp ./sampling/tf_sampling_g.cu.o -o ./sampling/tf_sampling_so.so -I $CUDA_ROOT/include -L $CUDA_ROOT/lib64/ -fPIC   -I/home/guo/anaconda3/envs/tensorflow_gpuenv/lib/python3.8/site-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=0 -L/home/guo/anaconda3/envs/tensorflow_gpuenv/lib/python3.8/site-packages/tensorflow -l:libtensorflow_framework.so.2
echo "Sampling op compiled."

echo "All ops compiled successfully."

