
// ICPMeshAlignMFCDlg.h : header file
//

#pragma once

#include "OpenGLControl.h"

// CICPMeshAlignMFCDlg dialog
class CICPMeshAlignMFCDlg : public CDialog
{
// Construction
public:
	CICPMeshAlignMFCDlg(CWnd* pParent = NULL);	// standard constructor

	COpenGLControl m_oglWindow;

// Dialog Data
	enum { IDD = IDD_ICPMeshAlignMFC_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support


// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg void OnTimer(UINT_PTR nIDEvent);
	afx_msg void OnBnClickedAlignmesh();
	afx_msg void OnBnClickedLoad();
	afx_msg void OnBnClickedAlign();
	afx_msg void OnBnClickedNoise();
	afx_msg void OnBnClickedRotatex();
	afx_msg void OnBnClickedRotatey();
	afx_msg void OnBnClickedRotatez();
};
