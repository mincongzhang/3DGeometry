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
double eps = 0.0;				// error bound
istream* dataIn = NULL;		// input for data points
istream* queryIn = NULL;	// input for query points

//Assign mesh points to ANN point array
void getsampledAnnArray(size_t sample_Pts,int sample_ratio,MyMesh &mesh,ANNpointArray &dataArray)
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
void MeshAlign(MyMesh &source_mesh, MyMesh &target_mesh)
{
	/*ANN kd-tree find nearest point*/
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

	int max_sourcePts = source_mesh.n_vertices();					//max source points
	int max_targetPts = target_mesh.n_vertices();					//max target points
	int sample_ratio = 20;									//sample ratio
	int sample_sourcePts = max_sourcePts/sample_ratio;		//sample source points
	int sample_targetPts = max_targetPts/sample_ratio;		//sample target points

	double mean_dist = 0.0;    //mean distance 
	double thresh = 0.000005;	   //threshold for distance judgement

	sampledSourceArray = annAllocPts(sample_sourcePts, dim);
	sampledTargetArray = annAllocPts(sample_targetPts, dim);
	MatchArray		   = annAllocPts(sample_sourcePts, dim);

	//assign sampled meshes to ANN array,directly modify the address of arrays
	getsampledAnnArray(sample_sourcePts,sample_ratio,source_mesh,sampledSourceArray);
	getsampledAnnArray(sample_sourcePts,sample_ratio,target_mesh,sampledTargetArray);

	//build kd-tree
	kdTree = new ANNkd_tree(	// build search structure
		sampledTargetArray,		// the data points
		sample_targetPts,		// number of points
		dim);					// dimension of space

	/*MATRICES CALCULATION*/
	//Q MATRIX(target mesh) and P MATRIX (source mesh)
	//matching source points in kd-tree and assign coordinates to array
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
		matrix index 
		[0 1 2
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

	/*
	TODO inverse p*(p_transpose) cross product array
	(ignore this will not affect the result too much)
	for(int d=0;d<9;d++)
	{
	here to inverse ppt_cross_array[d];
	}

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
	*/

	//cross product matrix 
	/*
	gsl_matrix_alloc (size_t n1, size_t n2) : matrix of size n1 rows by n2 columns
	matrix index 
	[0 1 2
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
			//ignore the inverse of ppt_cross_array, will not affect the result
			gsl_matrix_set (cross_matrix, row, col,qpt_cross_array[id]);   
			id++;
		}
	}

	//rotation matrix
	double rotation [9];

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

	//rotation matrix: R* = VV*UT
	cblas_dgemm(CblasRowMajor, 
		CblasNoTrans, CblasNoTrans, 3, 3, 3,
		1.0, U_array, 3,VT_array , 3, 0.0, rotation, 3);

	//Translation matrix: Q_mean - R*Pmean
	double trans [3];
	double RP_mean[3];
	cblas_dgemm(CblasRowMajor, 
		CblasNoTrans, CblasNoTrans, 3, 1, 3,
		1.0, rotation, 3, P_mean, 1, 0.0, RP_mean, 1);

	for(int d=0;d<dim;d++)
	{
		trans[d] = Q_mean[d]-RP_mean[d];
	}

	//Rotate and translate source data points to a new mesh
	for (auto it = source_mesh.vertices_begin(); it != source_mesh.vertices_end(); ++it)
	{
		double oldPt [3] = {};
		double newPt [3] = {};
		for(int d=0;d<dim;d++){
			newPt[d]=trans[d];
			oldPt[d]=*(source_mesh.point(it).data()+d);
		}

		//Mutiply by rotation matrix
		cblas_dgemm(CblasRowMajor, 
			CblasNoTrans, CblasNoTrans, 3, 1, 3,
			1.0, rotation, 3,oldPt, 1, 1, newPt, 1);

		//update the source mesh
		for(int d=0;d<dim;d++){
			*(source_mesh.point(it).data()+d)=float(newPt[d]);
		}
	}

};

void RotateMesh(double rotate_theta,MyMesh &mesh)
{
	//double theta = 10*2*M_PI/360;
	double rotation [9] = {};
	double trans [3] = {};
	double Pt_mean [3] = {};
	double Pt_sum [3] = {};
	int mesh1size = mesh.n_vertices();

	switch(ROTATE_CONTROL)
	{
	case 1:
		//rotate at x axis
		rotation[0] = 1.0;
		rotation[4] = cos(rotate_theta);
		rotation[5] = -sin(rotate_theta);
		rotation[7] = sin(rotate_theta);
		rotation[8] = cos(rotate_theta);
		break;
	case 2:
		//rotate at y axis
		rotation[0] = cos(rotate_theta);
		rotation[2] = sin(rotate_theta);
		rotation[4] = 1.0;
		rotation[6] = -sin(rotate_theta);
		rotation[8] = cos(rotate_theta);
		break;
	case 3:
		//rotate at z axis
		rotation[0] = cos(rotate_theta);
		rotation[1] = -sin(rotate_theta);
		rotation[3] = sin(rotate_theta);
		rotation[4] = cos(rotate_theta);
		rotation[8] = 1.0;
		break;
	}

	//get sum and mean for current mesh
	//so that to get Translation array
	for (auto it = mesh.vertices_begin(); it != mesh.vertices_end(); ++it)
	{
		double Pt [3] = {};
		for(int d=0;d<dim;d++)
		{
			Pt[d]+=*(mesh.point(it).data()+d);
			Pt_sum[d] += Pt[d]; 
		}
	}

	for(int d=0;d<dim;d++)
	{
		Pt_mean[d] =  Pt_sum[d]/mesh1size;
		//get Translation array
		trans[d] = Pt_mean[d];
	}


	//rotation along the center of current mesh
	for (auto it = mesh.vertices_begin(); it != mesh.vertices_end(); ++it)
	{
		double oldPt [3] = {};
		double newPt [3] = {};
		for(int d=0;d<dim;d++)
		{
			oldPt[d]=*(mesh.point(it).data()+d);
			//Translate to origin
			oldPt[d]-=trans[d];
		}

		//Mutiply by rotation matrix
		cblas_dgemm(CblasRowMajor, 
			CblasNoTrans, CblasNoTrans, 3, 1, 3,
			1.0, rotation, 3,oldPt, 1, 1, newPt, 1);

		//update the source mesh
		for(int d=0;d<dim;d++)
		{
			//Translate back
			newPt[d]+=trans[d];
			*(mesh.point(it).data()+d)=float(newPt[d]);
		}
	}
	ROTATE_CONTROL = 0;
}

void AddNoise(double noise_standard_deviation,MyMesh &mesh)
{
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0,noise_standard_deviation); //Gaussian distribution: mean value = 0.0

	for (auto it = mesh.vertices_begin(); it != mesh.vertices_end(); ++it)
	{
		double Pt[3] = {};
		for (int d=0;d<dim;d++)
		{
			Pt[d]=*(mesh.point(it).data()+d);
			double randn = distribution(generator);
			if ((randn>=-1.0)&&(randn<=1.0))							        //Gaussian distribution range [-1.0,1.0]
			{
				Pt[d]= Pt[d]*(1.0+randn);
				*(mesh.point(it).data()+d)=float(Pt[d]);
			}
		}
	}
	NOISE_CONTROL = false;
}

double getArea(MyMesh::Point neighbourPt1,MyMesh::Point neighbourPt2,MyMesh::Point currentPt)
{
	//Area = sqrt(p*(p-a)*(p-b)*(p-c)),p = (a+b+c)/2
	double area,a,b,c,p;
	a = sqrt(pow((neighbourPt1.data()[0]-neighbourPt2.data()[0]),2)
		+pow((neighbourPt1.data()[1]-neighbourPt2.data()[1]),2)
		+pow((neighbourPt1.data()[2]-neighbourPt2.data()[2]),2));

	b = sqrt(pow((neighbourPt1.data()[0]-currentPt.data()[0]),2)
		+pow((neighbourPt1.data()[1]-currentPt.data()[1]),2)
		+pow((neighbourPt1.data()[2]-currentPt.data()[2]),2));

	c = sqrt(pow((neighbourPt2.data()[0]-currentPt.data()[0]),2)
		+pow((neighbourPt2.data()[1]-currentPt.data()[1]),2)
		+pow((neighbourPt2.data()[2]-currentPt.data()[2]),2));

	p = (a+b+c)/2;

	area = sqrt(p*(p-a)*(p-b)*(p-c));

	return area;
}




void FindNeighbours(int neighbour_num,ANNkd_tree* kdTree,ANNpointArray meshArray,ANNpoint Pt,
					std::vector<MyMesh::Point> &neighbours,vector<double> &distVector,double &radius)
{
	ANNidxArray		nnIdx;					// near neighbor indices
	ANNdistArray	dists;					// near neighbor distances
	nnIdx = new ANNidx[neighbour_num];		// allocate near neigh indices
	dists = new ANNdist[neighbour_num];		// allocate near neighbor dists

	kdTree->annkSearch(			// search
		Pt,						// query point
		neighbour_num,	        // number of near neighbors
		nnIdx,					// nearest neighbors (returned)
		dists,					// distance (returned)
		eps);					// error bound

	//radius = max(dist)
	for (int i=0;i<neighbour_num;i++)
	{
		if(*(dists+i)>radius)
		{
			radius = *(dists+i); 
		}
	}

	radius /= 0.9; //reject outlying points 

	for(int i=0;i<neighbour_num;i++)
	{
		if(*(dists+i)<radius)
		{
			ANNpoint current_neighbour;
			MyMesh::Point  neighbourPt;
			current_neighbour = annAllocPt(dim);

			current_neighbour = meshArray[*(nnIdx+i)];
			neighbourPt.data()[0] = double(current_neighbour[0]);
			neighbourPt.data()[1] = double(current_neighbour[1]);
			neighbourPt.data()[2] = double(current_neighbour[2]);
			neighbours.push_back(neighbourPt);
			distVector.push_back(*(dists+i));
		}
	}

	delete [] nnIdx; 
	delete [] dists;
}


/*FIVE MESH DENOISING FUNCTIONS*/


/*Get normal vectors from KNN
-> bilateral filter denoising of vertices*/
void Denoise1(MyMesh &mesh)
{
	/*ANN kd-tree find nearest point*/
	ANNpointArray	meshArray;				// mesh points array
	ANNpoint		Pt;						// point
	ANNkd_tree*		kdTree;					// search structure

	int PtNum = mesh.n_vertices();
	meshArray = annAllocPts(PtNum, dim);
	Pt = annAllocPt(dim);

	//assign mesh points to ANN array
	for (auto it = mesh.vertices_begin(); it != mesh.vertices_end(); ++it)
	{   
		//Pt get the space of data array
		double getPt[3] = {};

		//Pt get the coordinates of mesh point
		int index = it->idx();
		Pt = meshArray[index];
		for(int d = 0;d < dim; d++)
		{
			getPt[d] = *(mesh.point(it).data()+d);
			Pt[d] = getPt[d];
		}
		//assign Pt coordinates to data array
		meshArray[index] = Pt;
	}

	//build kd-tree
	kdTree = new ANNkd_tree(	// build search structure
		meshArray,				// the data points
		PtNum,					// number of points
		dim);					// dimension of space

	//calculate normal for each vertex and save in normal mesh 
	MyMesh normal_mesh = mesh;
	int normal_neighbour_num = 6;
	for (auto it = mesh.vertices_begin(); it != mesh.vertices_end(); ++it)
	{   
		MyMesh::Point neighbourPt1;
		MyMesh::Point neighbourPt2;
		MyMesh::Point currentPt;
		currentPt = mesh.point(it);

		//Pt get the coordinates from mesh array
		int index = it->idx();
		Pt = meshArray[index];

		std::vector<MyMesh::Point> norm_neighbours;
		vector<double> distVector;
		//initial radius for reject neighbours outside the range
		double normal_radius = 0.0;

		//Find K(neighbours for normal calculation) nearest neighbours, 
		//return norm_neighbours,distVector,normal_radius
		FindNeighbours(normal_neighbour_num,kdTree,meshArray,Pt,
			norm_neighbours,distVector,normal_radius);

		//calculate normal
		double normal[3] = {};
		double normal_length = 0.0;
		double weight_sum = 0.0;
		int normal_neighbour_size = norm_neighbours.size();

		for(int i=0;i<normal_neighbour_size;i++)
		{
			if(i!=(normal_neighbour_size-1))
			{
				neighbourPt1 = norm_neighbours.at(i);
				neighbourPt2 = norm_neighbours.at(i+1);
			}
			else
			{
				neighbourPt1 = norm_neighbours.at(i);
				neighbourPt2 = norm_neighbours.at(0);
			}

			//calculate area of the triangle for the weight of normal
			//Area = sqrt(p*(p-a)*(p-b)*(p-c)),p = (a+b+c)/2
			double weight;
			weight = getArea(neighbourPt1,neighbourPt2,currentPt);
			weight_sum += weight;
			//cross product to get normal
			normal[0] += weight*(neighbourPt1.data()[1]-currentPt.data()[1])*(neighbourPt2.data()[2]-currentPt.data()[2])
				-(neighbourPt1.data()[2]-currentPt.data()[2])*(neighbourPt2.data()[1]-currentPt.data()[1]);
			normal[1] += weight*(neighbourPt1.data()[2]-currentPt.data()[2])*(neighbourPt2.data()[0]-currentPt.data()[0])
				-(neighbourPt1.data()[0]-currentPt.data()[0])*(neighbourPt2.data()[2]-currentPt.data()[2]);
			normal[2] += weight*(neighbourPt1.data()[0]-currentPt.data()[0])*(neighbourPt2.data()[1]-currentPt.data()[1])
				-(neighbourPt1.data()[1]-currentPt.data()[1])*(neighbourPt2.data()[0]-currentPt.data()[0]);
		}

		//mean normal 
		normal[0] = normal[0]/weight_sum;
		normal[1] = normal[1]/weight_sum;
		normal[2] = normal[2]/weight_sum;

		//get length
		normal_length = sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);

		//normalize normal
		normal[0] = normal[0]/normal_length;
		normal[1] = normal[1]/normal_length;
		normal[2] = normal[2]/normal_length;

		//assign to normal_mesh
		for(int d=0;d<dim;d++){
			*(normal_mesh.point(it).data()+d)=float(normal[d]);
		}
	}// end of for (auto it = mesh.vertices_begin(); it != mesh.vertices_end(); ++it)

	//initial parameters for denoising
	double radius = 0.0;
	double sigma_c = 0.05; //depend on the distance of the points
	double sigma_s = 0.0;
	int neighbour_num = 12;

	//get certain neighbour and denoise
	for (auto it = mesh.vertices_begin(); it != mesh.vertices_end(); ++it)
	{   
		//Pt get the coordinates from mesh array
		int index = it->idx();
		Pt = meshArray[index];
		std::vector<MyMesh::Point> neighbours;
		vector<double> distVector;

		//Find K(neighbours for normal calculation) nearest neighbours, 
		//return neighbours,distVector,radius
		FindNeighbours(neighbour_num,kdTree,meshArray,Pt,
			neighbours,distVector,radius);

		//get height from current vertex to tangent plane
		//update neighbour number
		neighbour_num = neighbours.size();
		vector<double> height;
		double mean_height = 0.0;
		for(int i=0;i<neighbour_num;i++)
		{
			MyMesh::Point  current_neighbour;
			current_neighbour = neighbours.at(i);

			//get normal vector from normal mesh
			double normal_vector[3]={};
			normal_vector[0]= *normal_mesh.point(it).data();
			normal_vector[1]= *(normal_mesh.point(it).data()+1);
			normal_vector[2]= *(normal_mesh.point(it).data()+2);

			//calculate height
			height.push_back((Pt[0]-current_neighbour.data()[0])*normal_vector[0]+(Pt[1]-current_neighbour.data()[1])*normal_vector[1]+(Pt[2]-current_neighbour.data()[2])*normal_vector[2]);
			mean_height += height.at(i);
		}

		//Calculate standard deviation(sigma_s)
		mean_height /= neighbour_num;
		for(int i=0;i<neighbour_num;i++)
		{
			sigma_s +=  pow((height.at(i)-mean_height),2);
		}
		sigma_s = sqrt(sigma_s);

		//Bilateral Mesh Denoising
		double sum = 0;
		double normalizer = 0;
		double t,Wc,Ws;

		for(int i=0;i<neighbour_num;i++)
		{
			//get t
			t = distVector.at(i);

			//get Wc, Ws, sum, normalizer
			Wc = exp(-t*t/(2*sigma_c*sigma_c));
			Ws = exp(-height.at(i)*height.at(i)/(2*sigma_s*sigma_s));
			sum += (Wc*Ws)*height.at(i);
			normalizer += Wc*Ws;
		}

		//assign back to original mesh
		for(int d=0;d<dim;d++){
			//new_v = v-n*(sum/normalizer)
			*(mesh.point(it).data()+d) += -(*(normal_mesh.point(it).data()+d))*float(sum/normalizer);
		}

	}// end of for (auto it = mesh.vertices_begin(); it != mesh.vertices_end(); ++it)

	// clean kd-tree
	delete kdTree;
	annClose(); 

	DENOISE1_CONTROL = false;
}

/*Get normal vectors from one-ring neighbours by OpenMesh library 
-> bilateral filter denoising of vertices*/
void Denoise2(MyMesh &mesh)
{
	/*ANN kd-tree find nearest point*/
	ANNpointArray   meshArray;				// mesh points array
	ANNpoint		Pt;						// point
	ANNkd_tree*		kdTree;					// search structure
	int				PtNum = mesh.n_vertices();

	Pt		  = annAllocPt(dim);
	meshArray = annAllocPts(PtNum, dim);

	//Find neighbour points for calculating normal vectors
	MyMesh::Point  neighbourPt1;
	MyMesh::Point  neighbourPt2;

	//calculate normal for each vertex and save in normal mesh 
	MyMesh normal_mesh = mesh;
	std::vector<MyMesh::Point> one_ring_neighbours;
	MyMesh::Point              currentPt;

	//assign mesh points to ANN array, and calculate normal for each vertex
	for (auto it = mesh.vertices_begin(); it != mesh.vertices_end(); ++it)
	{   
		/*assign mesh points to ANN array*/

		int index = it->idx();
		//Pt get the space of data array
		Pt = meshArray[index];

		//Pt get the coordinates of mesh point
		double getPt[3] = {};
		for(int d = 0;d < dim; d++)
		{
			getPt[d] = *(mesh.point(it).data()+d);
			Pt[d] = getPt[d];
		}
		//assign Pt coordinates to data array
		meshArray[index] = Pt;


		//Pt get the coordinates from mesh array
		currentPt = mesh.point(it);

		//Find one-ring neighbours
		MyMesh::VertexIter          v_it;
		MyMesh::VertexVertexIter    vv_it;
		v_it = it;
		for (vv_it=mesh.vv_iter( v_it ); vv_it; ++vv_it)
		{
			one_ring_neighbours.push_back(mesh.point( vv_it ));
		}

		/*calculate normal vector*/
		double normal[3] = {};
		double normal_length = 0.0;
		double weight_sum = 0.0;
		int one_ring_neighboursize = one_ring_neighbours.size();

		for(int i=0;i<one_ring_neighboursize;i++)
		{
			if(i!=(one_ring_neighboursize-1))
			{
				neighbourPt1 = one_ring_neighbours.at(i);
				neighbourPt2 = one_ring_neighbours.at(i+1);
			}
			else
			{
				neighbourPt1 = one_ring_neighbours.at(i);
				neighbourPt2 = one_ring_neighbours.at(0);
			}

			//calculate area of the triangle for the weight of normal
			//Area = sqrt(p*(p-a)*(p-b)*(p-c)),p = (a+b+c)/2
			double weight;
			weight = getArea(neighbourPt1,neighbourPt2,currentPt);
			weight_sum += weight;
			//cross product to get normal
			normal[0] += weight*(neighbourPt1.data()[1]-currentPt.data()[1])*(neighbourPt2.data()[2]-currentPt.data()[2])
				-(neighbourPt1.data()[2]-currentPt.data()[2])*(neighbourPt2.data()[1]-currentPt.data()[1]);
			normal[1] += weight*(neighbourPt1.data()[2]-currentPt.data()[2])*(neighbourPt2.data()[0]-currentPt.data()[0])
				-(neighbourPt1.data()[0]-currentPt.data()[0])*(neighbourPt2.data()[2]-currentPt.data()[2]);
			normal[2] += weight*(neighbourPt1.data()[0]-currentPt.data()[0])*(neighbourPt2.data()[1]-currentPt.data()[1])
				-(neighbourPt1.data()[1]-currentPt.data()[1])*(neighbourPt2.data()[0]-currentPt.data()[0]);
		}

		//mean normal 
		normal[0] = normal[0]/weight_sum;
		normal[1] = normal[1]/weight_sum;
		normal[2] = normal[2]/weight_sum;

		//get length
		normal_length = sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);

		//normalize normal
		normal[0] = normal[0]/normal_length;
		normal[1] = normal[1]/normal_length;
		normal[2] = normal[2]/normal_length;

		//assign to normal_mesh
		for(int d=0;d<dim;d++){
			*(normal_mesh.point(it).data()+d)=float(normal[d]);
		}
	}// end of for (auto it = mesh.vertices_begin(); it != mesh.vertices_end(); ++it)

	//build kd-tree of the original mesh
	kdTree = new ANNkd_tree(	// build search structure
		meshArray,				// the data points
		PtNum,					// number of points
		dim);					// dimension of space

	//initial parameters for denoising
	double radius = 0.0;
	double sigma_c = 0.05; //depend on the distance of the points
	double sigma_s = 0.0;

	//get certain neighbour and denoise
	int neighbour_num = 12;
	for (auto it = mesh.vertices_begin(); it != mesh.vertices_end(); ++it)
	{   
		//Pt get the coordinates from mesh array
		int index = it->idx();
		Pt = meshArray[index];
		std::vector<MyMesh::Point> neighbours;
		vector<double> distVector;

		//Find K(neighbours for normal calculation) nearest neighbours, 
		//return neighbours,distVector,radius
		FindNeighbours(neighbour_num,kdTree,meshArray,Pt,
			neighbours,distVector,radius);

		//get height from current vertex to tangent plane
		//update neighbour number
		neighbour_num = neighbours.size();
		vector<double> height;
		double mean_height = 0.0;
		for(int i=0;i<neighbour_num;i++)
		{
			MyMesh::Point  current_neighbour;
			current_neighbour = neighbours.at(i);

			//get normal vector from normal mesh
			double normal_vector[3]={};
			normal_vector[0]= *normal_mesh.point(it).data();
			normal_vector[1]= *(normal_mesh.point(it).data()+1);
			normal_vector[2]= *(normal_mesh.point(it).data()+2);

			//calculate height
			height.push_back((Pt[0]-current_neighbour.data()[0])*normal_vector[0]+(Pt[1]-current_neighbour.data()[1])*normal_vector[1]+(Pt[2]-current_neighbour.data()[2])*normal_vector[2]);
			mean_height += height.at(i);
		}

		//Calculate standard deviation(sigma_s)
		mean_height /= neighbour_num;
		for(int i=0;i<neighbour_num;i++)
		{
			sigma_s +=  pow((height.at(i)-mean_height),2);
		}
		sigma_s = sqrt(sigma_s);

		//Bilateral Mesh Denoising
		double sum = 0.0;
		double normalizer = 0.0;
		double t,Wc,Ws;

		for(int i=0;i<neighbour_num;i++)
		{
			//get t
			t = distVector.at(i);

			//get Wc, Ws, sum, normalizer
			Wc = exp(-t*t/(2*sigma_c*sigma_c));
			Ws = exp(-height.at(i)*height.at(i)/(2*sigma_s*sigma_s));
			sum += (Wc*Ws)*height.at(i);
			normalizer += Wc*Ws;
		}

		//assign back to original mesh
		for(int d=0;d<dim;d++){
			//new_v = v-n*(sum/normalizer)
			*(mesh.point(it).data()+d) += -(*(normal_mesh.point(it).data()+d))*float(sum/normalizer);
		}
	}// end of for (auto it = mesh.vertices_begin(); it != mesh.vertices_end(); ++it)

	// clean kd-tree
	delete kdTree;
	annClose(); 

	DENOISE2_CONTROL = false;
}

/*Get normal vectors from OpenMesh library
-> bilateral filter denoising of vertices*/
void Denoise3(MyMesh &mesh)
{
	/*ANN kd-tree find nearest point*/
	ANNpointArray	meshArray;				// mesh points array
	ANNpoint		Pt;						// point
	ANNkd_tree*		kdTree;					// search structure

	int PtNum = mesh.n_vertices();
	meshArray = annAllocPts(PtNum, dim);
	Pt = annAllocPt(dim);

	//assign mesh points to ANN array
	for (auto it = mesh.vertices_begin(); it != mesh.vertices_end(); ++it)
	{   
		//Pt get the space of data array
		double getPt[3] = {};

		//Pt get the coordinates of mesh point
		int index = it->idx();
		Pt = meshArray[index];
		for(int d = 0;d < dim; d++)
		{
			getPt[d] = *(mesh.point(it).data()+d);
			Pt[d] = getPt[d];
		}
		//assign Pt coordinates to data array
		meshArray[index] = Pt;
	}

	//build kd-tree
	kdTree = new ANNkd_tree(	// build search structure
		meshArray,				// the data points
		PtNum,					// number of points
		dim);					// dimension of space

	/*Request vertex normal*/
	// request vertex normals, so the mesh reader can use normal information
	// if available
	mesh.request_vertex_normals();

	OpenMesh::IO::Options opt;

	// If the file did not provide vertex normals, then calculate them
	if ( !opt.check( OpenMesh::IO::Options::VertexNormal ) )
	{
		// we need face normals to update the vertex normals
		mesh.request_face_normals();
		// let the mesh update the normals
		mesh.update_normals();

		// dispose the face normals, as we don't need them anymore
		mesh.release_face_normals();
	}

	/*Bilateral filtering*/
	//initial parameters for denoising
	double radius = 0.0;
	double sigma_c = 0.05; //depend on the distance of the points
	double sigma_s = 0.0;
	int neighbour_num = 12;

	//get certain neighbour and denoise
	for (auto it = mesh.vertices_begin(); it != mesh.vertices_end(); ++it)
	{   
		//Pt get the coordinates from mesh array
		int index = it->idx();
		Pt = meshArray[index];
		std::vector<MyMesh::Point> neighbours;
		vector<double> distVector;

		//Find K(neighbours for normal calculation) nearest neighbours, 
		//return neighbours,distVector,radius
		FindNeighbours(neighbour_num,kdTree,meshArray,Pt,
			neighbours,distVector,radius);

		//get height from current vertex to tangent plane
		//update neighbour number
		neighbour_num = neighbours.size();
		vector<double> height;
		double mean_height = 0.0;
		for(int i=0;i<neighbour_num;i++)
		{
			MyMesh::Point  current_neighbour;
			current_neighbour = neighbours.at(i);

			//get normal vector from normal mesh
			double normal_vector[3]={};
			normal_vector[0]= *mesh.normal(it).data();
			normal_vector[1]= *(mesh.normal(it).data()+1);
			normal_vector[2]= *(mesh.normal(it).data()+2);

			//calculate height
			height.push_back((Pt[0]-current_neighbour.data()[0])*normal_vector[0]+(Pt[1]-current_neighbour.data()[1])*normal_vector[1]+(Pt[2]-current_neighbour.data()[2])*normal_vector[2]);
			mean_height += height.at(i);
		}

		//Calculate standard deviation(sigma_s)
		mean_height /= neighbour_num;
		for(int i=0;i<neighbour_num;i++)
		{
			sigma_s +=  pow((height.at(i)-mean_height),2);
		}
		sigma_s = sqrt(sigma_s);

		//Bilateral Mesh Denoising
		double sum = 0;
		double normalizer = 0;
		double t,Wc,Ws;

		for(int i=0;i<neighbour_num;i++)
		{
			//get t
			t = distVector.at(i);

			//get Wc, Ws, sum, normalizer
			Wc = exp(-t*t/(2*sigma_c*sigma_c));
			Ws = exp(-height.at(i)*height.at(i)/(2*sigma_s*sigma_s));
			sum += (Wc*Ws)*height.at(i);
			normalizer += Wc*Ws;
		}

		//assign back to original mesh
		for(int d=0;d<dim;d++){
			//new_v = v-n*(sum/normalizer)
			*(mesh.point(it).data()+d) += -(*(mesh.normal(it).data()+d))*float(sum/normalizer);
		}

	}// end of for (auto it = mesh.vertices_begin(); it != mesh.vertices_end(); ++it)

	// clean kd-tree
	delete kdTree;
	annClose(); 

	DENOISE3_CONTROL = false;
}

/*Uniform Laplacian smoothing*/
void LaplaceDenoise(MyMesh &mesh)
{
	//Find neighbour points for calculating normal vectors
	MyMesh::Point  neighbourPt1;
	MyMesh::Point  neighbourPt2;

	//calculate normal for each vertex 
	std::vector<MyMesh::Point> neighbours;
	MyMesh::Point               currentPt;
	for (auto it = mesh.vertices_begin(); it != mesh.vertices_end(); ++it)
	{   
		//Find one-ring neighbours
		MyMesh::VertexIter          v_it;
		MyMesh::VertexVertexIter    vv_it;
		v_it = it;
		for (vv_it=mesh.vv_iter(v_it); vv_it; ++vv_it)
		{
			neighbours.push_back(mesh.point(vv_it));
		}

		int neighbours_size = neighbours.size();

		//define a scale factor to make gradual change
		double scale_factor = 0.01; 

		//assign to normal_mesh
		for(int d=0;d<dim;d++)
		{
			double laplace_vector = 0.0;
			for(int i=0;i<neighbours_size;i++)
			{
				laplace_vector += neighbours.at(i).data()[d];// - currentPt.data()[d];
			}
			laplace_vector /= double(neighbours_size);

			*(mesh.point(it).data()+d) -= float(scale_factor*laplace_vector);
		}
	}// end of for (auto it = mesh.vertices_begin(); it != mesh.vertices_end(); ++it)

	LAPLACE_DENOISE_CONTROL = false;
}


/*Get normal vectors from OpenMesh library
-> bilateral filter denoising of normals*/
void BiNormDenoise(MyMesh &mesh)
{
	//request face normals
	mesh.request_face_normals();
	// let the mesh update the normals
	mesh.update_normals();

	/*Bilateral Filtering of Face Normal*/

	//initial parameters for denoising
	double sigma_c = 10.95; //depend on the distance of the points
	double sigma_s = 0.0;

	/*step1: Update face normal*/
	for (MyMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it)
	{
		//MyMesh::FaceVertexIter fv_it;
		//MyMesh::FaceFaceIter ff_it;
		double d_mean = 0.0;
		vector<double> d_vector;

		//calculate the sigma s
		for(MyMesh::FaceFaceIter ff_it=mesh.ff_iter(f_it);ff_it;++ff_it) //find the one-ring face neighbours	
		{
			double d;
			//get the face normal
			d = mesh.normal(ff_it).data()[0];

			//normal difference projected on the surface normal at the selected point
			// d = ni*(ni-nj)
			d = mesh.normal(f_it).data()[0]*(mesh.normal(f_it).data()[0]-mesh.normal(ff_it).data()[0])
				+mesh.normal(f_it).data()[1]*(mesh.normal(f_it).data()[1]-mesh.normal(ff_it).data()[1])
				+mesh.normal(f_it).data()[2]*(mesh.normal(f_it).data()[2]-mesh.normal(ff_it).data()[2]);
			//for calculate the mean of d
			d_vector.push_back(d);
			d_mean += d;
		}//end of for(ff_it=mesh.ff_iter(f_it);ff_it;++ff_it)
		d_mean/=d_vector.size();

		for (int i=0;i<d_vector.size();i++)
		{
			sigma_s += pow(d_vector.at(i)-d_mean,2);
		}
		sigma_s = sqrt(sigma_s);

		//Initialise the parameters
		double normalizer = 0;
		double Wc, Ws, normal_diff, centroid_diff;
		MyMesh::Point centroid_i,centroid_j;
		MyMesh::Point sum;
		sum.data()[0] = 0.0;
		sum.data()[1] = 0.0;
		sum.data()[2] = 0.0;
		//find the one-ring face neighbours		
		for(MyMesh::FaceFaceIter ff_it=mesh.ff_iter(f_it);ff_it;++ff_it)
		{
			//neighbour face
			//get the face normal

			//normal difference projected on the surface normal at the selected point
			// normal_diff = ni*(ni-nj)
			normal_diff = mesh.normal(f_it).data()[0]*(mesh.normal(f_it).data()[0]-mesh.normal(ff_it).data()[0])
				+mesh.normal(f_it).data()[1]*(mesh.normal(f_it).data()[1]-mesh.normal(ff_it).data()[1])
				+mesh.normal(f_it).data()[2]*(mesh.normal(f_it).data()[2]-mesh.normal(ff_it).data()[2]);

			//get the centroid of face i and j
			mesh.calc_face_centroid(f_it,centroid_i);
			mesh.calc_face_centroid(ff_it,centroid_j);

			Ws = exp(-normal_diff*normal_diff/(2*sigma_s*sigma_s));

			centroid_diff = sqrt(pow(centroid_j.data()[0]-centroid_i.data()[0],2)
				+pow(centroid_j.data()[1]-centroid_i.data()[1],2)
				+pow(centroid_j.data()[2]-centroid_i.data()[2],2));

			Wc = exp(-centroid_diff*centroid_diff/(2*sigma_c*sigma_c));

			sum.data()[0] = sum.data()[0]+(Wc*Ws)*mesh.normal(ff_it).data()[0];
			sum.data()[1] = sum.data()[1]+(Wc*Ws)*mesh.normal(ff_it).data()[1];
			sum.data()[2] = sum.data()[2]+(Wc*Ws)*mesh.normal(ff_it).data()[2];

			normalizer += Wc*Ws;
		}//end of for(ff_it=mesh.ff_iter(f_it);ff_it;++ff_it)
		sum.data()[0] = sum.data()[0]/normalizer;
		sum.data()[1] = sum.data()[1]/normalizer;
		sum.data()[2] = sum.data()[2]/normalizer;

		mesh.set_normal(f_it,sum);
	}//end of for (MyMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it)

	/*step2: Least Square Error*/
	for(MyMesh::VertexIter v_it = mesh.vertices_begin();v_it!=mesh.vertices_end();++v_it)
	{
		double sum_0 = 0.0;
		double sum_1 = 0.0;
		double sum_2 = 0.0;
		for(MyMesh::VertexIHalfedgeIter vih_it=mesh.vih_begin(v_it.handle());vih_it!=mesh.vih_end(v_it.handle());++vih_it)
		{
			MyMesh::HalfedgeHandle startHEH,endHEH;
			MyMesh::VertexHandle startVH,endVH;
			MyMesh::FaceHandle startFH,endFH;
			double diff_0,diff_1,diff_2;

			startHEH = vih_it.handle();
			endHEH = mesh.opposite_halfedge_handle(startHEH);
			MyMesh::FaceIter f_it;

			if (!mesh.is_boundary(startHEH) && !mesh.is_boundary(endHEH))
			{
				startVH = mesh.to_vertex_handle(startHEH);
				endVH = mesh.to_vertex_handle(endHEH);

				startFH = mesh.opposite_face_handle(endHEH);
				endFH = mesh.opposite_face_handle(startHEH);

				diff_0 = mesh.point(endVH).data()[0] - mesh.point(startVH).data()[0];
				diff_1 = mesh.point(endVH).data()[1] - mesh.point(startVH).data()[1];
				diff_2 = mesh.point(endVH).data()[2] - mesh.point(startVH).data()[2];

				//*f_it = startFH.idx;
				//sum_0 += diff_0*mesh.normal(startFH).data()[0]*mesh.normal(startFH).data()[0]
				//+ diff_1*mesh.normal(startFH).data()[0]*mesh.normal(startFH).data()[1]
				//+ diff_2*mesh.normal(startFH).data()[0]*mesh.normal(startFH).data()[2]
				//+ diff_0*mesh.normal(endFH).data()[0]*mesh.normal(endFH).data()[0]
				//+ diff_1*mesh.normal(endFH).data()[0]*mesh.normal(endFH).data()[1]
				//+ diff_2*mesh.normal(endFH).data()[0]*mesh.normal(endFH).data()[2];

				//sum_1 += diff_0*mesh.normal(startFH).data()[0]*mesh.normal(startFH).data()[1]
				//+ diff_1*mesh.normal(startFH).data()[1]*mesh.normal(startFH).data()[1]
				//+ diff_2*mesh.normal(startFH).data()[1]*mesh.normal(startFH).data()[2]
				//+ diff_0*mesh.normal(endFH).data()[0]*mesh.normal(endFH).data()[1]
				//+ diff_1*mesh.normal(endFH).data()[1]*mesh.normal(endFH).data()[1]
				//+ diff_2*mesh.normal(endFH).data()[1]*mesh.normal(endFH).data()[2];

				//sum_2 += diff_0*mesh.normal(startFH).data()[0]*mesh.normal(startFH).data()[2]
				//+ diff_1*mesh.normal(startFH).data()[1]*mesh.normal(startFH).data()[1]
				//+ diff_2*mesh.normal(startFH).data()[2]*mesh.normal(startFH).data()[2]
				//+ diff_0*mesh.normal(endFH).data()[0]*mesh.normal(endFH).data()[2]
				//+ diff_1*mesh.normal(endFH).data()[1]*mesh.normal(endFH).data()[1]
				//+ diff_2*mesh.normal(endFH).data()[2]*mesh.normal(endFH).data()[2];

				sum_0 += diff_0*mesh.normal(startFH).data()[0]*mesh.normal(startFH).data()[0]
						+diff_0*mesh.normal(endFH).data()[0]*mesh.normal(endFH).data()[0];
				sum_1 += diff_1*mesh.normal(startFH).data()[1]*mesh.normal(startFH).data()[1]
						+diff_1*mesh.normal(endFH).data()[1]*mesh.normal(endFH).data()[1];
				sum_2 += diff_2*mesh.normal(startFH).data()[2]*mesh.normal(startFH).data()[2]
						+diff_2*mesh.normal(endFH).data()[2]*mesh.normal(endFH).data()[2];

				double lambda = 0.01;

				*(mesh.point(v_it).data()) += float(lambda*sum_0);
				*(mesh.point(v_it).data()+1) += float(lambda*sum_1);
				*(mesh.point(v_it).data()+2) += float(lambda*sum_2);
			}// end of if (!mesh.is_boundary(startHEH) && !mesh.is_boundary(endHEH))
		}//end of for(MyMesh::VertexIHalfedgeIter vih_it=mesh.vih_begin(v_it.handle());vih_it!=mesh.vih_end(v_it.handle());++vih_it)
	}//end of for(MyMesh::VertexIter v_it = mesh.vertices_begin();v_it!=mesh.vertices_end();++v_it)

	BINORM_DENOISE_CONTROL = false;
	mesh.release_face_normals();
}
