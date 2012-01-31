######################################################################
# Automatically generated by qmake (2.01a) Mon Jan 30 20:47:26 2012
######################################################################

QT += opengl
TEMPLATE = app
TARGET = qt4_mandelbrot
INCLUDEPATH += .
CONFIG += x86

# Input
HEADERS += appglwidget.h globals.h qt4_mandelbrot.h simplePBO.h
FORMS += qt4_mandelbrot.ui
SOURCES += appglwidget.cpp main.cpp qt4_mandelbrot.cpp simplePBO.cpp

CUDA_SOURCES += kernelPBO.cu

OTHER_FILES += $$CUDA_SOURCES

########################################################################
#  CUDA
########################################################################
macx {
  # auto-detect CUDA path
  CUDA_DIR = $$system(which nvcc | sed 's,/bin/nvcc$,,')
  INCLUDEPATH += $$CUDA_DIR/include
  QMAKE_LIBDIR += $$CUDA_DIR/lib
  LIBS += -lcudart

  cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.obj
  cuda.commands = nvcc -c -Xcompiler $$join(QMAKE_CXXFLAGS,",") $$join(INCLUDEPATH,'" -I "','-I "','"') ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
  cuda.dependcy_type = TYPE_C
  cuda.depend_command = nvcc -M -Xcompiler $$join(QMAKE_CXXFLAGS,",") $$join(INCLUDEPATH,'" -I "','-I "','"') ${QMAKE_FILE_NAME} | sed "s,^.*: ,," | sed "s,^ *,," | tr -d '\\n'
}
cuda.input = CUDA_SOURCES
QMAKE_EXTRA_COMPILERS += cuda
########################################################################