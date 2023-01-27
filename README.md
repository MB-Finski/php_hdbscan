# PHP-HDBSCAN

A HDBSCAN implementation written in PHP. Builds loosely on top of the Rubix ML library. Implements a dual-tree kNN for core distance estimation and a dual-tree Borůvka for building the minimum spanning tree. Designed primarily for use with the Recognize -app for Nextcloud. 

The code is extremely rough around the edges, sorry about that!

## Features
- Beats the Rubix DBSCAN in speed by about 1/3rd.
- Much more robust clustering
- Dataset size doesn't affect cluster stability as with DBSCAN
- Support for exporting and storing core distances for (slightly) accelerating consequtive clusterings.

## Versions:
27.1.2023: v0.1-alpha of HDBSCAN -- core functionality implemented and working

## References

March W.B., Ram P., Gray A.G.
Fast Euclidean Minimum Spanning Tree: Algorithm, Analysis, and Applications
Proc. ACM SIGKDD’10, 2010, 603-611, https://mlpack.org/papers/emst.pdf

Curtin, R., March, W., Ram, P., Anderson, D., Gray, A., & Isbell, C. (2013, May). 
Tree-independent dual-tree algorithms. 
In International Conference on Machine Learning (pp. 1435-1443). PMLR.


