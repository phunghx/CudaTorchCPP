extern "C"
{
     void dummy(int* ptrFromLua, int size);  // Declare a function (function prototype)
}
void dummy(int* ptrFromLua, int size) {
  int i;
  for (i = 0; i < size; i++)
    ptrFromLua[i] = i + 1;
  int j;
  j=10;
  return;
}

