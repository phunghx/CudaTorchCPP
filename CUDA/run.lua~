ffi = require("ffi")
-- Load myLib
myLib = ffi.load(paths.cwd() .. '/libtest3.so')
-- Function prototypes definition
ffi.cdef [[
   int my_test_func2();
]]
myLib.my_test_func2()
