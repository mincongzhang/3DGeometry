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
void getsampledAnnArray(MyMesh &mesh,ANNpointArray &dataArray, size_t sample_Pts, int sample_ratio);
void RotateMesh(MyMesh &mesh);
void AddNoise(MyMesh &mesh2);