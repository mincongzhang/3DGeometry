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

void MeshAlign(MyMesh &mesh1, MyMesh &mesh2);
void getsampledAnnArray(size_t sample_Pts,int sample_ratio,MyMesh &mesh,ANNpointArray &dataArray);
void RotateMesh(double rotate_theta,MyMesh &mesh);
void AddNoise(double noise_standard_deviation,MyMesh &mesh2);