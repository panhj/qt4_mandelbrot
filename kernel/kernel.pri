CUDA_SOURCES = $$PWD/kernelPBO.cu

OTHER_FILES += $$CUDA_SOURCES

include(../cuda_compiler.pri)
