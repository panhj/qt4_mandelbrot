
#include "appglwidget.h"

#include <QString>
#include <QMessageBox>

//-----------------------------------------------------------------------------
// AppGLWidget
//-----------------------------------------------------------------------------
AppGLWidget::AppGLWidget(QWidget *parent)
    : QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
{
    QGLFormat fmt;
    fmt.setVersion(2, 0);
    QGLFormat::setDefaultFormat(fmt);

    strFrames = "Frames: 0";
    frames    = 0;
    tiupdate  = new QTimer( this );
    tifps     = new QTimer( this );
    connect ( tiupdate, SIGNAL( timeout() ), this, SLOT ( update() ) );
    connect ( tifps,    SIGNAL( timeout() ), this, SLOT ( framesCount() ) );
    tifps->start(1000);
    timeElapse.start();
    glError = GL_NO_ERROR;
    qDebug() << QGLFormat::openGLVersionFlags();
}
//-----------------------------------------------------------------------------
// AppGLWidget
//-----------------------------------------------------------------------------
AppGLWidget::~AppGLWidget()
{
    delete tiupdate;
    delete tifps;
}
//-----------------------------------------------------------------------------
// initializeGL
//-----------------------------------------------------------------------------
void AppGLWidget::initializeGL()
{
    int vmaj = format().majorVersion();
    int vmin = format().minorVersion();
    if( vmaj < 2 ){
        QMessageBox::warning(this,
                             tr("Wrong OpenGL version"),
                             tr("OpenGL version 2.0 or higher needed. You have %1.%2, so some functions may not work properly.").arg(vmaj).arg(vmin));
    }
    qDebug() << tr("OpenGL Version: %1.%2").arg(vmaj).arg(vmin);

    spbo.initCuda();

    glClearColor(0, 0, 0, 1);
    glDisable(GL_DEPTH_TEST);
    glShadeModel(GL_FLAT);
    glDisable(GL_LIGHTING);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    startUpdateTimer();
}
//-----------------------------------------------------------------------------
// resizeGL
//-----------------------------------------------------------------------------
void AppGLWidget::resizeGL(int w, int h){
    glViewport(0, 0, w, h);
    glEnable(GL_TEXTURE_2D);
    spbo.resize(w, h);
    glMatrixMode(GL_PROJECTION); //Select The Projection Matrix
    glLoadIdentity(); //Reset The Projection Matrix

    glMatrixMode(GL_MODELVIEW); //Select The Modelview Matrix
    glLoadIdentity(); //Reset The Modelview Matrix
}
//-----------------------------------------------------------------------------
// paintGL
//-----------------------------------------------------------------------------
void AppGLWidget::paintGL(){
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW); //Select The Modelview Matrix
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    // run CUDA kernel
    spbo.runCuda( timeElapse.elapsed() );
    // now bind buffer after cuda is done
    spbo.bind();

    // Draw a single Quad with texture coordinates for each vertex.
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f,1.0f);  glVertex3f(-1.0f,-1.0f,-1.0f);
    glTexCoord2f(0.0f,0.0f);  glVertex3f(-1.0f,1.0f,-1.0f);
    glTexCoord2f(1.0f,0.0f);  glVertex3f(1.0f,1.0f,-1.0f);
    glTexCoord2f(1.0f,1.0f);  glVertex3f(1.0f,-1.0f,-1.0f);
    glEnd();

    spbo.release();

    glColor3f(0,1,0);
    renderText(10,10, strFrames);
    glColor3f(1,1,1);

    ++frames;

    glError = glGetError();
}
//-----------------------------------------------------------------------------
// startUpdateTimer
//-----------------------------------------------------------------------------
void AppGLWidget::startUpdateTimer()
{
    if( tiupdate->isActive() == false ){
        timeElapse.restart();
        tiupdate->start(20);
    }
}
//-----------------------------------------------------------------------------
// stopUpdateTimer
//-----------------------------------------------------------------------------
void AppGLWidget::stopUpdateTimer()
{
    tiupdate->stop();
    timeElapse.restart();
}
//-----------------------------------------------------------------------------
// framesCount
//-----------------------------------------------------------------------------
void AppGLWidget::framesCount(){
    strFrames.setNum(frames);
    strFrames.prepend(tr("Frames: "));
    frames=0;
}
//-----------------------------------------------------------------------------
// update - SLOT Timer tiupdate
//-----------------------------------------------------------------------------
void AppGLWidget::update(){
    if( glError != GL_NO_ERROR ){ // OpenGL ocurred
        stopUpdateTimer();

        QMessageBox::warning(this, "OpenGL Error: "+QString::number(glError), getGLError());
        glError=GL_NO_ERROR;
    }else{
        updateGL();
    }
}
//-----------------------------------------------------------------------------
// getGLError
//-----------------------------------------------------------------------------
QString AppGLWidget::getGLError(){
    QString exception = "No error";
    switch (glError)
    {
    // see macro on top
    GLERROR(GL_INVALID_ENUM)
            GLERROR(GL_INVALID_VALUE)
            GLERROR(GL_INVALID_OPERATION)
            GLERROR(GL_STACK_OVERFLOW)
            GLERROR(GL_STACK_UNDERFLOW)
            GLERROR(GL_OUT_OF_MEMORY)
        #ifdef GL_INVALID_INDEX
            GLERROR(GL_INVALID_INDEX)
        #endif
        #ifdef GL_INVALID_FRAMEBUFFER_OPERATION_EXT
            GLERROR(GL_INVALID_FRAMEBUFFER_OPERATION_EXT)
        #endif
            default:
        exception.sprintf("Unknown GL error: %04x\n", glError);
    break;
    }
    return exception;
}
