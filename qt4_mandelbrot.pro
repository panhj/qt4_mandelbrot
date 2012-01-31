TEMPLATE = app
QT = gui core network opengl
DESTDIR = bin
macx:CONFIG += x86

DESTDIR = bin
OBJECTS_DIR = bld/o
MOC_DIR = bld/moc
UI_DIR = bld/ui
RCC_DIR = bld/rcc

include(src/src.pri)
include(ui/ui.pri)
include(kernel/kernel.pri)
