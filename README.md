OpenMesh
========

OpenMesh practices

1.ICP
ANN, GSL and OpenMesh libraries are used to achieve the ICP meshes alignment algorithm. 
With ANN library, kd-tree is built to find nearest neighbour points. Matrices computing is available in GSL library.
UI is built in MFC. (A bug is found when degree 180 bunny is used as the 2nd mesh)

2.Denoising
Five different approaches are implemented for the mesh denoising[1];
Denoise1 - Apply bilateral filter to vertices, using KNN for normal calculation[1];
Denoise2 - Apply bilateral filter to vertices, using one-ring neighbour for normal calculation[1];
Denoise3 - Apply bilateral filter to vertices, directly using vertices normal from OpenMesh library;
LaplaceDenoise - Apply uniform Laplacian filter to denoise;
BiNormDenoise - Apply bilateral filter to face normal vectors, and then update each vertex in faces[2].


[1]Fleishman, Shachar, Iddo Drori, and Daniel Cohen-Or. "Bilateral mesh denoising." ACM Transactions on Graphics (TOG). Vol. 22. No. 3. ACM, 2003.
[2]Lee, Kai-Wah, and Wen-Ping Wang. "Feature-preserving mesh denoising via bilateral normal filtering." Computer Aided Design and Computer Graphics, 2005. Ninth International Conference on. IEEE, 2005.

P.S.
In paper [1], the psudocode Vertex v' = v+n*(sum/normalizer) should be changed into Vertex v' = v-n*(sum/normalizer);
In paper [2], the equation(6) and (7) from LSE computation, n*trans(n) should be changed into trans(n)*n;
otherwise the results are not correct.
