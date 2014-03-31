#include "stdafx.h"
#include "OpenGLControl.h"
#include ".\openglcontrol.h"
#include "MeshOperation.h"

#include <math.h>
#include <stdio.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <ANN/ANN.h>

using namespace std; // make std:: accessible

//kd tree
// Global variables
//
int k = 1;					// number of nearest neighbors
int dim = 3;				// dimension
double eps = 0;				// error bound
istream* dataIn = NULL;		// input for data points
istream* queryIn = NULL;	// input for query points


//Align meshes
void MeshAlign(MyMesh &mesh1, MyMesh &mesh2)
{
	/*ANN kd-tree find nearest point*/
	ANNpointArray	//TODO;		// source data points array
	ANNpointArray	//TODO;		// target data points array
	ANNpointArray	//TODO;				// matched data points array
	ANNpoint		//TODO;				// query point
	ANNpoint		//TODO;				// match point
	ANNidxArray		nnIdx;					// near neighbor indices
	ANNdistArray	dists;					// near neighbor distances
	ANNkd_tree*		kdTree;					// search structure

	//TODO annAllocPt(dim) to ANN point (query point and match point)
	
	nnIdx = new ANNidx[k];					// allocate near neigh indices
	dists = new ANNdist[k];					// allocate near neighbor dists

	//TODO				//max source points
	//TODO				//max target points
	//TODO				//sample ratio
	//TODO				//sample source points
	//TODO				//sample target points


	//TODO annAllocPts(sample_sourcePts, dim); (source,target,match arrays)

	//assign sampled meshes to ANN array,directly modify the address of arrays
	//TODO write a sample function

	//build kd-tree
	kdTree = new ANNkd_tree(	// build search structure
		//TODO,		// the data points
		//TODO,		// number of points
		dim);					// dimension of space

	/*MATRICES CALCULATION*/
	//Q MATRIX(target mesh) and P MATRIX (source mesh)
	//matching source points in kd-tree and assign coordinates to array
	//TODO

	//TODO calculate sum and mean
	
	// clean kd-tree
	delete [] nnIdx; 
	delete [] dists;
	delete kdTree;
	annClose(); 

	//TODO get diff between coordinates and mean, and cross product

	//TODO
	//cross product matrix 
	/*
	gsl_matrix_alloc (size_t n1, size_t n2) : matrix of size n1 rows by n2 columns
	matrix index [0 1 2
	3 4 5
	6 7 8] 
	*/

	//TODO
	//rotation matrix

	//TODO
	//SVD = U S V^T


	//TODO RELEASE MATRICES
	//gsl_matrix_free ();

	//TODO
	//compute rotation matrix

	//TODO
	//Rotate source data points to a new mesh

	//update the source mesh
};
