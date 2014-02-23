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
#include <random>

using namespace std; // make std:: accessible

//kd tree
// Global variables
//
int k = 1;					// number of nearest neighbors
int dim = 3;				// dimension
double eps = 0;				// error bound
istream* dataIn = NULL;		// input for data points
istream* queryIn = NULL;	// input for query points

//Assign mesh points to ANN point array
void getsampledAnnArray(MyMesh &mesh,ANNpointArray &dataArray, size_t sample_Pts, int sample_ratio)
{
	ANNpoint Pt;
	Pt = annAllocPt(dim);

	double getPt[3] = {};
	//downsampling count
	int count = 0;

	for (auto it = mesh.vertices_begin(); it != mesh.vertices_end(); ++it)
	{   
		int index = it->idx();

		//downsampling
		if(index % sample_ratio == 0)
		{
			//Pt get the space of data array
			Pt = dataArray[count];
			//Pt get the coordinates of mesh point
			for(int d = 0;d < dim; d++)
			{
				getPt[d] = *(mesh.point(it).data()+d);
				Pt[d] = getPt[d];
			}
			//assign Pt coordinates to data array
			dataArray[count] = Pt;

			count++;
			if(count==sample_Pts) break;
		}
	}
};

//Align meshes
void MeshAlign(MyMesh &mesh1, MyMesh &mesh2)
{
	ANNpointArray	sampledSourceArray;		// source data points array
	ANNpointArray	sampledTargetArray;		// target data points array
	ANNpointArray	MatchArray;				// matched data points array
	ANNpoint		queryPt;				// query point
	ANNpoint		matchPt;				// match point
	ANNidxArray		nnIdx;					// near neighbor indices
	ANNdistArray	dists;					// near neighbor distances
	ANNkd_tree*		kdTree;					// search structure

	queryPt = annAllocPt(dim);
	matchPt = annAllocPt(dim);
	nnIdx = new ANNidx[k];					// allocate near neigh indices
	dists = new ANNdist[k];					// allocate near neighbor dists

	int max_sourcePts = mesh1.n_vertices();					//max source points
	int max_targetPts = mesh2.n_vertices();					//max target points
	int sample_ratio = 20;									//sample ratio
	int sample_sourcePts = max_sourcePts/sample_ratio;		//sample source points
	int sample_targetPts = max_targetPts/sample_ratio;		//sample target points

	double mean_dist = 0.0;    //mean distance 
	double thresh = 0.00001;	   //threshold for distance judgement

	sampledSourceArray = annAllocPts(sample_sourcePts, dim);
	sampledTargetArray = annAllocPts(sample_targetPts, dim);
	MatchArray		   = annAllocPts(sample_sourcePts, dim);

	//assign sampled meshes to ANN array,directly modify the address of arrays
	getsampledAnnArray(mesh1,sampledSourceArray,sample_sourcePts,sample_ratio);
	getsampledAnnArray(mesh2,sampledTargetArray,sample_targetPts,sample_ratio);

	//build kd-tree
	kdTree = new ANNkd_tree(	// build search structure
		sampledTargetArray,		// the data points
		sample_targetPts,		// number of points
		dim);					// dimension of space

	/*MATRICES CALCULATION*/

	//Q MATRIX and P MATRIX
	//matching source points in kd-tree
	//and assign coordinates to array
	double Q_sum  [3] = {};
	double P_sum  [3] = {};
	double Q_mean [3] = {};
	double P_mean [3] = {};

	for(int m=0;m <sample_sourcePts ; m++)
	{
		//Pt get the coordinates from source array
		queryPt=sampledSourceArray[m];

		kdTree->annkSearch(		// search
			queryPt,			// query point
			k,					// number of near neighbors
			nnIdx,				// nearest neighbors (returned)
			dists,				// distance (returned)
			eps);				// error bound

		//calculate the sum of distances
		mean_dist += *dists; 
		MatchArray[m]=sampledTargetArray[*nnIdx];

		//Pt get the matched coordinates from match array 
		matchPt = MatchArray[m];

		//get sum of each coordinate
		for(int d=0; d<dim; d++)
		{
			Q_sum[d] += matchPt[d];
			P_sum[d] += queryPt[d];
		}
	}

	//get mean distance for convergence
	mean_dist = mean_dist/sample_sourcePts;
	if (mean_dist<thresh) return;

	//get mean of each coordinate
	for(int d=0;d<dim;d++)
	{
		Q_mean [d] = Q_sum[d]/sample_sourcePts;
		P_mean [d] = P_sum[d]/sample_sourcePts;
	}

	// clean kd-tree
	delete [] nnIdx; 
	delete [] dists;
	delete kdTree;
	annClose(); 

	//get diff between coordinates and mean, and cross product
	double qpt_cross_array [9] = {};
	double ppt_cross_array [9] = {};
	for(int m=0;m <sample_sourcePts ; m++)
	{
		double Q_diff [3] ;
		double P_diff [3] ;
		queryPt = sampledSourceArray[m];
		matchPt = MatchArray[m];

		for(int d=0; d<dim; d++)
		{
			Q_diff[d] = matchPt[d]-Q_mean[d];
			P_diff[d] = queryPt[d]-P_mean[d];

		}
		/*   q*(p_transpose) cross product array
		matrix index [0 1 2
		3 4 5
		6 7 8] */
		qpt_cross_array[0] += Q_diff[0]*P_diff[0];
		qpt_cross_array[1] += Q_diff[0]*P_diff[1];
		qpt_cross_array[2] += Q_diff[0]*P_diff[2];
		qpt_cross_array[3] += Q_diff[1]*P_diff[0];
		qpt_cross_array[4] += Q_diff[1]*P_diff[1];
		qpt_cross_array[5] += Q_diff[1]*P_diff[2];
		qpt_cross_array[6] += Q_diff[2]*P_diff[0];
		qpt_cross_array[7] += Q_diff[2]*P_diff[1];
		qpt_cross_array[8] += Q_diff[2]*P_diff[2];

		/*   p*(p_transpose) cross product array
		matrix index [0 1 2
		3 4 5
		6 7 8] */
		ppt_cross_array[0] += P_diff[0]*P_diff[0];
		ppt_cross_array[1] += P_diff[0]*P_diff[1];
		ppt_cross_array[2] += P_diff[0]*P_diff[2];
		ppt_cross_array[3] += P_diff[1]*P_diff[0];
		ppt_cross_array[4] += P_diff[1]*P_diff[1];
		ppt_cross_array[5] += P_diff[1]*P_diff[2];
		ppt_cross_array[6] += P_diff[2]*P_diff[0];
		ppt_cross_array[7] += P_diff[2]*P_diff[1];
		ppt_cross_array[8] += P_diff[2]*P_diff[2];
	}

	//inverse p*(p_transpose) cross product array
	/*for(int d=0;d<9;d++)
	{
	ppt_cross_array[d] = 1.0f/ppt_cross_array[d];
	}*/

	//get an array of 
	//(product of q*(p_transpose) cross product matrix) by (inverse p*(p_transpose) cross product matrix)
	double A [9];
	A[0] = qpt_cross_array[0]*ppt_cross_array[0]+qpt_cross_array[1]*ppt_cross_array[3]+qpt_cross_array[2]*ppt_cross_array[6];
	A[1] = qpt_cross_array[0]*ppt_cross_array[1]+qpt_cross_array[1]*ppt_cross_array[4]+qpt_cross_array[2]*ppt_cross_array[7];
	A[2] = qpt_cross_array[0]*ppt_cross_array[2]+qpt_cross_array[1]*ppt_cross_array[5]+qpt_cross_array[2]*ppt_cross_array[8];
	A[3] = qpt_cross_array[3]*ppt_cross_array[0]+qpt_cross_array[4]*ppt_cross_array[3]+qpt_cross_array[5]*ppt_cross_array[6];
	A[4] = qpt_cross_array[3]*ppt_cross_array[1]+qpt_cross_array[4]*ppt_cross_array[4]+qpt_cross_array[5]*ppt_cross_array[7];
	A[5] = qpt_cross_array[3]*ppt_cross_array[2]+qpt_cross_array[4]*ppt_cross_array[5]+qpt_cross_array[5]*ppt_cross_array[8];
	A[6] = qpt_cross_array[6]*ppt_cross_array[0]+qpt_cross_array[7]*ppt_cross_array[3]+qpt_cross_array[8]*ppt_cross_array[6];
	A[7] = qpt_cross_array[6]*ppt_cross_array[1]+qpt_cross_array[7]*ppt_cross_array[4]+qpt_cross_array[8]*ppt_cross_array[7];
	A[8] = qpt_cross_array[6]*ppt_cross_array[2]+qpt_cross_array[7]*ppt_cross_array[5]+qpt_cross_array[8]*ppt_cross_array[8];

	//cross product matrix 
	/*
	gsl_matrix_alloc (size_t n1, size_t n2) : matrix of size n1 rows by n2 columns
	matrix index [0 1 2
	3 4 5
	6 7 8] 
	*/
	gsl_matrix * cross_matrix = gsl_matrix_alloc (dim, dim);
	unsigned int id=0;
	for (int row=0;row<dim;row++)
	{
		for(int col=0;col<dim; col++)
		{
			//gsl_matrix_set (cross_matrix, row, col,A[id]); 
			gsl_matrix_set (cross_matrix, row, col,qpt_cross_array[id]); 
			id++;
		}
	}

	//Rotation matrix
	double Rotation [9];

	//SVD = U S V^T
	gsl_matrix * V = gsl_matrix_alloc (dim, dim);
	gsl_vector * S = gsl_vector_alloc (dim);
	gsl_vector * work = gsl_vector_alloc (dim);

	//cross_matrix is replaced by U; V is V; S is Sigma
	gsl_linalg_SV_decomp(cross_matrix, V, S, work);

	double U_array[9];
	double VT_array[9];
	unsigned int id1 = 0;
	for(int row =0; row<3;row++){
		for(int col =0; col<3;col++)
		{
			//array U
			U_array[id1] = gsl_matrix_get(cross_matrix,row,col); 
			//array VT
			VT_array[id1] = gsl_matrix_get(V,col,row);
			id1++;
		}
	}

	//RELEASE MATRICES
	gsl_matrix_free (cross_matrix);
	gsl_matrix_free (V);
	gsl_vector_free (S);
	gsl_vector_free (work);

	//Rotation matrix: R* = VV*UT
	cblas_dgemm(CblasRowMajor, 
		CblasNoTrans, CblasNoTrans, 3, 3, 3,
		1.0, U_array, 3,VT_array , 3, 0.0, Rotation, 3);

	//Translation matrix: Q_mean - R*Pmean
	double Trans [3];
	double RP_mean[3];
	cblas_dgemm(CblasRowMajor, 
		CblasNoTrans, CblasNoTrans, 3, 1, 3,
		1.0, Rotation, 3, P_mean, 1, 0.0, RP_mean, 1);

	for(int d=0;d<dim;d++)
	{
		Trans[d] = Q_mean[d]-RP_mean[d];
	}

	//Rotate and translate source data points to a new mesh
	for (auto it = mesh1.vertices_begin(); it != mesh1.vertices_end(); ++it)
	{
		double oldPt [3] = {};
		double newPt [3] = {};
		for(int d=0;d<dim;d++){
			newPt[d]=Trans[d];
			oldPt[d]=*(mesh1.point(it).data()+d);
		}

		//Mutiply by Rotation matrix
		cblas_dgemm(CblasRowMajor, 
			CblasNoTrans, CblasNoTrans, 3, 1, 3,
			1.0, Rotation, 3,oldPt, 1, 1, newPt, 1);

		//update the source mesh
		for(int d=0;d<dim;d++){
			*(mesh1.point(it).data()+d)=float(newPt[d]);
		}
	}

};

void RotateMesh(MyMesh &mesh1)
{
	double theta = 0.1;
	double Rotation [9] = {};
	double Trans [3] = {};
	double Pt_mean [3] = {};
	double Pt_sum [3] = {};
	int mesh1size = mesh1.n_vertices();
 
	switch(ROTATE_CONTROL)
	{
	case 1:
		//rotate at x axis
		Rotation[0] = 1.0;
		Rotation[4] = cos(theta);
		Rotation[5] = -sin(theta);
		Rotation[7] = sin(theta);
		Rotation[8] = cos(theta);
		break;
	case 2:
		//rotate at y axis
		Rotation[0] = cos(theta);
		Rotation[2] = sin(theta);
		Rotation[4] = 1.0;
		Rotation[6] = -sin(theta);
		Rotation[8] = cos(theta);
		break;
	case 3:
		//rotate at z axis
		Rotation[0] = cos(theta);
		Rotation[1] = -sin(theta);
		Rotation[3] = sin(theta);
		Rotation[4] = cos(theta);
		Rotation[8] = 1.0;
		break;
	}

	//get sum and mean for current mesh
	//so that to get Translation array
	for (auto it = mesh1.vertices_begin(); it != mesh1.vertices_end(); ++it)
	{
		double Pt [3] = {};
		for(int d=0;d<dim;d++)
		{
			Pt[d]+=*(mesh1.point(it).data()+d);
			Pt_sum[d] += Pt[d]; 
		}
	}

	for(int d=0;d<dim;d++)
	{
		Pt_mean[d] =  Pt_sum[d]/mesh1size;
		//get Translation array
		Trans[d] = Pt_mean[d];
	}


	//Rotation along the center of current mesh
	for (auto it = mesh1.vertices_begin(); it != mesh1.vertices_end(); ++it)
	{
		double oldPt [3] = {};
		double newPt [3] = {};
		for(int d=0;d<dim;d++)
		{
			oldPt[d]=*(mesh1.point(it).data()+d);
			//Translate to origin
			oldPt[d]-=Trans[d];
		}

		//Mutiply by Rotation matrix
		cblas_dgemm(CblasRowMajor, 
			CblasNoTrans, CblasNoTrans, 3, 1, 3,
			1.0, Rotation, 3,oldPt, 1, 1, newPt, 1);

		//update the source mesh
		for(int d=0;d<dim;d++)
		{
			//Translate back
			newPt[d]+=Trans[d];
			*(mesh1.point(it).data()+d)=float(newPt[d]);
		}
	}
	ROTATE_CONTROL = 0;
}


void AddNoise(MyMesh &mesh2)
{
	double  stddev = 10.01;
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0,stddev);

	for (auto it = mesh2.vertices_begin(); it != mesh2.vertices_end(); ++it)
	{
		double Pt[3] = {};
		for (int d=0;d<dim;d++)
		{
			Pt[d]=*(mesh2.point(it).data()+d);
			double randn = distribution(generator);
			if ((randn>=-1.0)&&(randn<=1.0))
			{
				Pt[d]= Pt[d]*(1+randn);
				*(mesh2.point(it).data()+d)=float(Pt[d]);
			}
		}
	}
	NOISE_CONTROL = false;
}