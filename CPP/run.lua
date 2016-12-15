require('torch');
length=5e1;displace=true
ffi = require 'ffi'

ffi.cdef[[
	void testfunction();
]]
myLib = ffi.load('/data5/LuaExperiment/CPP/file1.so')
pf= function(...) print(string.format(...)) end

a=torch.IntTensor(length):random()
b=a:clone()
if display then
	print('Random vector')
	print(a)
end
timer = torch.Timer()
myLib['testfunction']()
---myLib.testfunction()

---myLib.dummy(torch.data(a), length)
---pf('C function %.2f ms', timer:time().real*1e3)

---if display then print(a) end

---timer = torch.Timer()
for i=1, length do b[i]=i end
pf('Lua loop %.2f ms', timer:time().real *1e3)
if display then print(b) end
