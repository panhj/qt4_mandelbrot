#ifndef QT4_MANDELBROT_H
#define QT4_MANDELBROT_H

#include <QtGui/QMainWindow>
#include "ui_qt4_mandelbrot.h"

class qt4_mandelbrot : public QMainWindow
{
    Q_OBJECT

public:
    qt4_mandelbrot(QWidget *parent = 0, Qt::WFlags flags = 0);
    ~qt4_mandelbrot();

private:
    Ui::qt4_mandelbrotClass ui;
};

#endif // QT4_MANDELBROT_H
