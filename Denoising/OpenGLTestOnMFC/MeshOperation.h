#pragma once
#include "afxwin.h"

#include <gl/gl.h>
#include <gl/glu.h>
#include <vector>
#include <ANN/ANN.h>

#undef min
#undef max
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
using namespace std;

typedef OpenMesh::PolyMesh_ArrayKernelT<>  MyMesh;

double getArea(MyMesh::Point neighbourPt1,MyMesh::Point neighbourPt2,MyMesh::Point currentPt);
void MeshAlign(MyMesh &target_mesh, MyMesh &source_mesh);
void getsampledAnnArray(size_t sample_Pts,int sample_ratio,MyMesh &mesh,ANNpointArray &dataArray);
void RotateMesh(double rotate_theta,MyMesh &mesh);
void AddNoise(double noise_standard_deviation,MyMesh &mesh);
void MarkOverlap(MyMesh &target_mesh, MyMesh &source_mesh,vector<bool> &overlap);
void FindNeighbours(int neighbour_num,ANNkd_tree* kdTree,ANNpointArray meshArray,ANNpoint Pt,
					std::vector<MyMesh::Point> &neighbours,vector<double> &distVector,double &radius);
void Denoise1(MyMesh &mesh);
void Denoise2(MyMesh &mesh);
void Denoise3(MyMesh &mesh);
void LaplaceDenoise(MyMesh &mesh);
void BiNormDenoise(MyMesh &mesh);
