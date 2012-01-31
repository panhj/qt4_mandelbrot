/******************************************************************************
**
** Author Matthias Werner
** EMail <wmatthias at users.sourceforge.net>
** Qt with CUDA and OpenGL Demonstration.
** Feel free to use and modify it for your own purposes.
** Have fun!
**
** Created and tested on 
** - Windows 7 Geforce 9800GT
** - CUDA Toolkit 4.0
** - OpenGL 2.1
** - Qt 4.7.4
** - Visual Studio 2010 C/C++ Compiler 
     (with Parallel Nsight 2.0 and Qt Visual Studio AddIn 1.1.9)
*******************************************************************************/

#ifndef __APPGLWIDGET_H
#define __APPGLWIDGET_H

// first 
#include "simplePBO.h"

#include <QtGui>
#include <QtOpenGL>
#include <QGLWidget>
#include <QTimer>

// for AppGLWidget::getGLError()
#ifndef GLERROR
#define GLERROR(e) case e: exception=#e; break;
#endif

/**
 * OpenGL frame widget.
 * Shows opengl scene being updated by frame timer.
 */
class AppGLWidget : public QGLWidget
{
  Q_OBJECT

  public:
    /**
     * Constructor.
     * @param parent Parent widget.
     */
    AppGLWidget(QWidget *parent);

    /**
     * Destructor (delete timers on heap).
     */
    ~AppGLWidget();

    /**
     * Starts Update Timer.
     * Timer is running at max speed 
        albeit frame rate may be capped by drivers (~60fps).
     */
    void startUpdateTimer();

    /// Stop the update timer.
    void stopUpdateTimer();

//-----------------------------------------------------------------------------
  public slots:
    /**
     * Update scene by calling updateGL(). 
     * If OpenGL Error ocurred, show it and do not update 
       (stopUpdateTimer).
     */
    void update();
    
  private slots:
    /**
     * Count frames and create debug string (strFrames). Reset frame counter.
     */
    void framesCount();

//-----------------------------------------------------------------------------
  private:
    /**
     * Initialize context, check for OpenGL driver and start Update Timer.
     */
    void initializeGL();

    /**
     * Resize OpenGL frame.
     * @param w Window width
     * @param h Window height
     */
    void resizeGL(int w, int h);

    /**
     * Paint OpenGL Frame.
     */
    void paintGL();

    /**
     * @return OpenGL error enum as QString.
     */
    QString getGLError();

  private:
    /// frame counter
    unsigned int frames;
    /// error opengl enum type
    GLenum  glError;
    /// time object for measuring elapsed times
    QTime   timeElapse;
    /// timer for updating frame (\see update())
    QTimer* tiupdate; 
    /// 1sec timer for counting frames
    QTimer* tifps; 
    /// string holding frames debug text
    QString strFrames; 
    /// SimplePBO object as pixel buffer object manager
    SimplePBO spbo;
    
};

#endif
