#include "stdafx.h"
#include "OpenGLControl.h"
#include ".\openglcontrol.h"
#include "MeshOperation.h"

#include <math.h>
#include <stdio.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <ANN/ANN.h>

using namespace std;					 // make std:: accessible

vector<MyMesh>  meshQueue;

bool ALIGN_CONTROL = false;
bool NOISE_CONTROL = false;
bool DENOISE1_CONTROL = false;
bool DENOISE2_CONTROL = false;
bool DENOISE3_CONTROL = false;
bool LAPLACE_DENOISE_CONTROL = false;
bool BINORM_DENOISE_CONTROL = false;
bool SHOWPOINTS_CONTROL = true;
bool SHOWFACES_CONTROL = false;
bool SHOWWIREFRAME_CONTROL = false;
int  ROTATE_CONTROL = 0;
double noise_standard_deviation = 0.01;  //standard_deviation for adding noise
double rotate_theta = 5*2*M_PI/360;		 //degree for rotating source mesh

	// Lighting components
	GLfloat  ambientLight[] = { 0.5f, 0.5f, 0.5f, 1.0f };
	GLfloat  diffuseLight[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	GLfloat  specular[] = { 1.0f, 1.0f, 1.0f, 1.0f};
	GLfloat	 lightPos[] = { 1.0f, 1.0f, 1.0f, 0.0f };

COpenGLControl::COpenGLControl(void)
{
	m_fPosX = 0.0f;						 // X position of model in camera view
	m_fPosY = -0.1f;					 // Y position of model in camera view
	m_fZoom = 1.0f;						 // Zoom on model in camera view
	m_fRotX = 0.0f;						 // Rotation on model in camera view
	m_fRotY	= 0.0f;						 // Rotation on model in camera view
	m_bIsMaximized = false;
}

COpenGLControl::~COpenGLControl(void)
{
}

BEGIN_MESSAGE_MAP(COpenGLControl, CWnd)
	ON_WM_PAINT()
	ON_WM_SIZE()
	ON_WM_CREATE()
	ON_WM_TIMER()
	ON_WM_MOUSEMOVE()
END_MESSAGE_MAP()

void COpenGLControl::OnPaint()
{

	//CPaintDC dc(this); // device context for painting
	ValidateRect(NULL);
}

void COpenGLControl::OnSize(UINT nType, int cx, int cy)
{
	CWnd::OnSize(nType, cx, cy);

	if (0 >= cx || 0 >= cy || nType == SIZE_MINIMIZED) return;

	// Map the OpenGL coordinates.
	glViewport(0, 0, cx, cy);

	// Projection view
	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();

	// Set our current view perspective
	gluPerspective(35.0f, (float)cx / (float)cy, 0.01f, 2000.0f);

	// Model view
	glMatrixMode(GL_MODELVIEW);

	switch (nType)
	{
		// If window resize token is "maximize"
	case SIZE_MAXIMIZED:
		{
			// Get the current window rect
			GetWindowRect(m_rect);

			// Move the window accordingly
			MoveWindow(6, 6, cx - 14, cy - 14);

			// Get the new window rect
			GetWindowRect(m_rect);

			// Store our old window as the new rect
			m_oldWindow = m_rect;

			break;
		}

		// If window resize token is "restore"
	case SIZE_RESTORED:
		{
			// If the window is currently maximized
			if (m_bIsMaximized)
			{
				// Get the current window rect
				GetWindowRect(m_rect);

				// Move the window accordingly (to our stored old window)
				MoveWindow(m_oldWindow.left, m_oldWindow.top - 18, m_originalRect.Width() - 4, m_originalRect.Height() - 4);

				// Get the new window rect
				GetWindowRect(m_rect);

				// Store our old window as the new rect
				m_oldWindow = m_rect;
			}

			break;
		}
	}
}

int COpenGLControl::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CWnd::OnCreate(lpCreateStruct) == -1) return -1;

	oglInitialize();

	return 0;
}

void COpenGLControl::OnDraw(CDC *pDC)
{
	// If the current view is perspective...
	glLoadIdentity();
	//glFrustum(-1, 1, -1, 1, 0.0, 40.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glTranslatef(0.0f, 0.0f, -m_fZoom);
	glTranslatef(m_fPosX, m_fPosY, 0.0f);
	glRotatef(m_fRotX, 1.0f, 0.0f, 0.0f);
	glRotatef(m_fRotY, 0.0f, 1.0f, 0.0f);
}

void COpenGLControl::OnTimer(UINT nIDEvent)
{
	switch (nIDEvent)
	{
	case 1:
		{
			// Clear color and depth buffer bits
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			// Draw OpenGL scene
			oglDrawScene();

			// Swap buffers
			SwapBuffers(hdc);

			break;
		}

	default:
		break;
	}

	CWnd::OnTimer(nIDEvent);
}

void COpenGLControl::OnMouseMove(UINT nFlags, CPoint point)
{
	int diffX = (int)(point.x - m_fLastX);
	int diffY = (int)(point.y - m_fLastY);
	m_fLastX  = (float)point.x;
	m_fLastY  = (float)point.y;

	// Left mouse button
	if (nFlags & MK_LBUTTON)
	{
		m_fRotX += (float)0.5f * diffY;

		if ((m_fRotX > 360.0f) || (m_fRotX < -360.0f))
		{
			m_fRotX = 0.0f;
		}

		m_fRotY += (float)0.5f * diffX;

		if ((m_fRotY > 360.0f) || (m_fRotY < -360.0f))
		{
			m_fRotY = 0.0f;
		}
	}

	// Middle mouse button
	else if (nFlags & MK_MBUTTON)
	{
		m_fZoom -= (float)0.01f * diffY;
	}

	// Right mouse button
	else if (nFlags & MK_RBUTTON)
	{
		m_fPosX += (float)0.0005f * diffX;
		m_fPosY -= (float)0.0005f * diffY;
	}

	OnDraw(NULL);

	CWnd::OnMouseMove(nFlags, point);
}

void COpenGLControl::oglCreate(CRect rect, CWnd *parent)
{
	CString className = AfxRegisterWndClass(CS_HREDRAW | CS_VREDRAW | CS_OWNDC, NULL, (HBRUSH)GetStockObject(BLACK_BRUSH), NULL);

	CreateEx(0, className, "OpenGL", WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS | WS_CLIPCHILDREN, rect, parent, 0);

	// Set initial variables' values
	m_oldWindow	   = rect;
	m_originalRect = rect;

	hWnd = parent;
}



void COpenGLControl::GLSetupRC()
{
	glEnable(GL_DEPTH_TEST);	// Hidden surface removal
	glEnable(GL_COLOR_MATERIAL);

	glEnable(GL_LIGHTING);
	glLightfv(GL_LIGHT0,GL_AMBIENT,ambientLight);
	glLightfv(GL_LIGHT0,GL_DIFFUSE,diffuseLight);
	glLightfv(GL_LIGHT0,GL_SPECULAR,specular);
	glLightfv(GL_LIGHT0,GL_POSITION,lightPos);
	glEnable(GL_LIGHT0);

	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glMaterialfv(GL_FRONT, GL_SPECULAR,specular);
	glMateriali(GL_FRONT,GL_SHININESS,120.0f);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f ); //background color
	// default color
	glColor3ub(0, 0, 255);
}


void COpenGLControl::oglInitialize(void)
{
	// Initial Setup:
	//
	static PIXELFORMATDESCRIPTOR pfd =
	{
		sizeof(PIXELFORMATDESCRIPTOR),
		1,
		PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,
		PFD_TYPE_RGBA,
		32, // bit depth
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		16, // z-buffer depth
		0, 0, 0, 0, 0, 0, 0,
	};

	// Get device context only once.
	hdc = GetDC()->m_hDC;

	// Pixel format.
	m_nPixelFormat = ChoosePixelFormat(hdc, &pfd);
	SetPixelFormat(hdc, m_nPixelFormat, &pfd);

	// Create the OpenGL Rendering Context.
	hrc = wglCreateContext(hdc);
	wglMakeCurrent(hdc, hrc);

	// Basic Setup:
	//
	GLSetupRC();
	// Set color to use when clearing the background.
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClearDepth(1.0f);
	////////////////////////////////////////

	///////////////////////////////////////
	// Turn on backface culling
	glFrontFace(GL_CCW);
	glCullFace(GL_BACK);

	// Turn on depth testing
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	// Send draw request
	OnDraw(NULL);


}

void COpenGLControl::oglDrawScene(void)
{
	//menu_system();

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	unsigned int meshsize = meshQueue.size();

	//new source mesh (mesh3) = rotated target mesh (mesh1)
	if(ROTATE_CONTROL!=0 && meshsize>=2)
	{
		RotateMesh(rotate_theta,meshQueue.at(meshsize-1));			
	};

	//add noise to source mesh (mesh2)
	if(NOISE_CONTROL && meshsize>=1)
	{
		AddNoise(noise_standard_deviation,meshQueue.at(meshsize-1));
	}

	//denoising
	if(DENOISE1_CONTROL && meshsize>=1)
	{
		Denoise1(meshQueue.at(meshsize-1));
	}
	if(DENOISE2_CONTROL && meshsize>=1)
	{
		Denoise2(meshQueue.at(meshsize-1));
	}
	if(DENOISE3_CONTROL && meshsize>=1)
	{
		Denoise3(meshQueue.at(meshsize-1));
	}
	if(LAPLACE_DENOISE_CONTROL && meshsize>=1)
	{
		LaplaceDenoise(meshQueue.at(meshsize-1));
	}
	if(BINORM_DENOISE_CONTROL && meshsize>=1)
	{
		BiNormDenoise(meshQueue.at(meshsize-1));
	}

	//align meshes
	if(ALIGN_CONTROL && meshsize>=2)
	{	
		//directly modify the address of each mesh
		for(int i=0;i< (meshsize-1);i++)
		{
			//align source mesh (mesh2) to target mesh (mesh1)
			MeshAlign(meshQueue.at(i+1),meshQueue.at(i));
		}
	}

	//draw meshes
	for (unsigned int i=0;i<meshsize;i++)
	{
		if(meshsize>0)
		{
			//change the colour for each mesh
			switch (i) 
			{
			case 0:
				glColor3f(GLfloat(0.8), GLfloat(0.8), GLfloat(0.2));
				break;
			case 1:
				glColor3f(GLfloat(0.7), GLfloat(0.5), GLfloat(1.0));
				break;
			case 2:
				glColor3f(GLfloat(0.6), GLfloat(1.0), GLfloat(0.5));
				break;
			default:
				glColor3f(GLfloat(0.5), GLfloat(0.5), GLfloat(0.5));
			};

			if(!SHOWPOINTS_CONTROL)
			{
				if(SHOWWIREFRAME_CONTROL)
				{
					glDisable(GL_LIGHTING);
					glBegin(GL_LINES);
				}
				else if(SHOWFACES_CONTROL)
				{
				glBegin(GL_POLYGON);
				}
				for (MyMesh::FaceIter it = meshQueue.at(i).faces_begin(); it != meshQueue.at(i).faces_end(); ++it)
				{
					MyMesh::FaceVertexIter fv_it;
					for (fv_it = meshQueue.at(i).fv_iter(it);fv_it;++fv_it)
					{
						MyMesh::VertexIter  v_it;
						MyMesh::VertexVertexIter    vv_it;
						vv_it= meshQueue.at(i).vv_iter( fv_it );
						glVertex3f(meshQueue.at(i).point(vv_it).data()[0],meshQueue.at(i).point(vv_it).data()[1],meshQueue.at(i).point(vv_it).data()[2]);
					}
				}
				glEnd();
			}
			else if(SHOWPOINTS_CONTROL)
			{
				glDisable(GL_LIGHTING);
				glPointSize(2.0);
				glBegin(GL_POINTS);
				for (auto it = meshQueue.at(i).vertices_begin(); it != meshQueue.at(i).vertices_end(); ++it)
				{

					int index = it->idx();
					auto point = meshQueue.at(i).point(it.handle());
					glVertex3f(point.data()[0],point.data()[1],point.data()[2]);
				}
				glEnd();
			}

		}
	}//end for (unsigned int i=0;i<meshsize;i++)
}
