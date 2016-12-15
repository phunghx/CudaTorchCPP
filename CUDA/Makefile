#testmain : main.cpp  libtest.so
#	g++ -c main.cpp
#	g++ -o testmain  -L.  -ldl -Wl,-rpath,.   -ltest -L/usr/local/cuda/lib64 -lcudart main.o

#libtest.so : link.o
#	g++  -shared -Wl,-soname,libtest.so -o libtest.so    test1.o test2.o link.o  -L/usr/local/cuda/lib64  -lcudart
#link.o : test1.cu test2.cu   test1.h test2.h
#	nvcc  -m64   -arch=sm_20 -dc  -Xcompiler '-fPIC' -Wno-deprecated-gpu-targets test1.cu test2.cu
#	nvcc  -m64   -arch=sm_20 -Xcompiler '-fPIC' -dlink -Wno-deprecated-gpu-targets test1.o test2.o -o link.o

libtest3.so : test2_link.o
	g++  -shared -o libtest3.so   test2_link.o -L/usr/local/cuda/lib64  -lcudart  -lcudadevrt -fPIC
test2_link.o : test2.o
        nvcc -arch=sm_35 -dlink -Xcompiler '-fPIC' -o test2_link.o test2.o -lcudadevrt -lcudart
test2.o: test2.cu
	nvcc  -m64   -arch=sm_35 -dc  -Xcompiler '-fPIC' -Wno-deprecated-gpu-targets test2.cu
clean :
	rm -f *.o *.so
