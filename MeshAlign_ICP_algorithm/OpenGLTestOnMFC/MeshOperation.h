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

void MeshAlign(MyMesh &target_mesh, MyMesh &source_mesh);
void getsampledAnnArray(size_t sample_Pts,int sample_ratio,MyMesh &mesh,ANNpointArray &dataArray);
void RotateMesh(double rotate_theta,MyMesh &mesh);
void AddNoise(double noise_standard_deviation,MyMesh &mesh);
void MarkOverlap(MyMesh &target_mesh, MyMesh &source_mesh,vector<bool> &overlap);