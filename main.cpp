#include "qt4_mandelbrot.h"
#include <QtGui/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    qt4_mandelbrot w;
    w.show();
    return a.exec();
}
