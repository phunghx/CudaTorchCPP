LIBOPTS = -shared
CFLAGS = -fPIC
CC = g++

libasdf.so : asdf.cpp
	$(CC) $< $(LIBOPTS) $(CFLAGS) -o $@

clean :
	rm *.so
