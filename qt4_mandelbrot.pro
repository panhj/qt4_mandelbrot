TEMPLATE = app
QT = gui core network opengl
DESTDIR = bin
macx:CONFIG += x86

include(src/src.pri)
include(ui/ui.pri)
include(kernel/kernel.pri)

