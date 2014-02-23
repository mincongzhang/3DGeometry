
// ICPMeshAlignMFC.cpp : Defines the class behaviors for the application.
//

#include "stdafx.h"
#include "ICPMeshAlignMFC.h"
#include "ICPMeshAlignMFCDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CICPMeshAlignMFCApp

BEGIN_MESSAGE_MAP(CICPMeshAlignMFCApp, CWinAppEx)
	ON_COMMAND(ID_HELP, CWinAppEx::OnContextHelp)
END_MESSAGE_MAP()

// CICPMeshAlignMFCApp construction

CICPMeshAlignMFCApp::CICPMeshAlignMFCApp()
{
	
	// Place all significant initialization in InitInstance
}


// The one and only CICPMeshAlignMFCApp object

CICPMeshAlignMFCApp theApp;


// CICPMeshAlignMFCApp initialization

BOOL CICPMeshAlignMFCApp::InitInstance()
{
	// InitCommonControlsEx() is required on Windows XP if an application
	// manifest specifies use of ComCtl32.dll version 6 or later to enable
	// visual styles.  Otherwise, any window creation will fail.
	INITCOMMONCONTROLSEX InitCtrls;
	InitCtrls.dwSize = sizeof(InitCtrls);


	InitCtrls.dwICC = ICC_WIN95_CLASSES;
	InitCommonControlsEx(&InitCtrls);

	CWinAppEx::InitInstance();

	AfxEnableControlContainer();

	// Standard initialization
	SetRegistryKey(_T("Local AppWizard-Generated Applications"));

	CICPMeshAlignMFCDlg dlg;
	m_pMainWnd = &dlg;
	INT_PTR nResponse = dlg.DoModal();
	if (nResponse == IDOK)
	{
		
		//  dismissed with OK
	}
	else if (nResponse == IDCANCEL)
	{
		
		//  dismissed with Cancel
	}

	// Since the dialog has been closed, return FALSE so that we exit the
	//  application, rather than start the application's message pump.
	return FALSE;
}
