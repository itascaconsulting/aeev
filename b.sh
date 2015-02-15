python setup.py build
c:/MinGW32-xy/bin/dllwrap.exe -mno-cygwin -mdll -static --output-lib build/temp.win32-2.7/Release/libaeev.a --def build/temp.win32-2.7/Release/aeev.def -s build/temp.win32-2.7/Release/aeev.o -Lc:/Python27/libs -Lc:/Python27/PCbuild -lpython27 -lmsvcr90 -o build/lib.win32-2.7/aeev.pyd

cp build/lib.win32-2.7/aeev.pyd .
