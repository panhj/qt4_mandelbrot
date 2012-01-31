########################################################################
#  CUDA
########################################################################
LIBS += -lcudart

cuda.name = Cuda ${QMAKE_FILE_IN}

macx {
  # auto-detect CUDA path
  CUDA_DIR = $$system(which nvcc | sed 's,/bin/nvcc$,,')
  INCLUDEPATH += $$CUDA_DIR/include
  QMAKE_LIBDIR += $$CUDA_DIR/lib
  LIBS += -lcudart

  cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.obj
  cuda.commands = nvcc -c -Xcompiler $$join(QMAKE_CXXFLAGS,",") $$join(INCLUDEPATH,'" -I "','-I "','"') ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
  cuda.dependency_type = TYPE_C
  cuda.depend_command = nvcc -M -Xcompiler $$join(QMAKE_CXXFLAGS,",") $$join(INCLUDEPATH,'" -I "','-I "','"') ${QMAKE_FILE_NAME} | sed '"'s,^.*: ,,'"' | sed '"'s,^ *,,'"' | tr -d '\\n'
}

win32 {
        LIBS += -LC:\\PROGRA~1\\NVIDIA~2\\CUDA\\v3.2\\lib\\Win32
        QMAKE_CUC = $(CUDA_BIN_PATH)/nvcc.exe

        cuda.CONFIG += no_link
        cuda.variable_out = OBJECTS

        isEmpty(CU_DIR):CU_DIR = .
        isEmpty(QMAKE_CPP_MOD_CU):QMAKE_CPP_MOD_CU =
        isEmpty(QMAKE_EXT_CPP_CU):QMAKE_EXT_CPP_CU = .cu

        INCLUDEPATH += $(CUDA_INC_PATH)

        QMAKE_CUFLAGS += $$QMAKE_CXXFLAGS
        QMAKE_CUFLAGS -= -Zc:wchar_t-
        DebugBuild:QMAKE_CUFLAGS += $$QMAKE_CXXFLAGS_DEBUG
        ReleaseBuild:QMAKE_CUFLAGS += $$QMAKE_CXXFLAGS_RELEASE
        QMAKE_CUFLAGS += $$QMAKE_CXXFLAGS_RTTI_ON $$QMAKE_CXXFLAGS_WARN_ON $$QMAKE_CXXFLAGS_STL_ON

        QMAKE_CUEXTRAFLAGS += -Xcompiler $$join(QMAKE_CUFLAGS, ",") $$CUFLAGS
        QMAKE_CUEXTRAFLAGS += $(DEFINES) $(INCPATH) $$join(QMAKE_COMPILER_DEFINES, " -D", -D)
        QMAKE_CUEXTRAFLAGS += -c
        QMAKE_EXTRA_VARIABLES += QMAKE_CUEXTRAFLAGS

        cuda.commands = $$QMAKE_CUC $(EXPORT_QMAKE_CUEXTRAFLAGS) -o $$OBJECTS_DIR/$${QMAKE_CPP_MOD_CU}${QMAKE_FILE_BASE}$${QMAKE_EXT_OBJ} ${QMAKE_FILE_NAME}
        cuda.output = $$OBJECTS_DIR/$${QMAKE_CPP_MOD_CU}${QMAKE_FILE_BASE}$${QMAKE_EXT_OBJ}
        silent:cuda.commands = @echo nvcc ${QMAKE_FILE_IN} && $$cuda.commands

        build_pass|isEmpty(BUILDS):cuclean.depends = compiler_cu_clean
        else:cuclean.CONFIG += recursive
        QMAKE_EXTRA_TARGETS += cuclean
}

cuda.input = CUDA_SOURCES
QMAKE_EXTRA_COMPILERS += cuda
