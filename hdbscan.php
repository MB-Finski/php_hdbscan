<?php

namespace OCA\Recognize\Service;

use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Graph\Nodes\Hypersphere;
use Rubix\ML\Graph\Nodes\Clique;
use Rubix\ML\Graph\Nodes\Ball;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\DataType;
use Rubix\ML\Helpers\Stats;
use Rubix\ML\Exceptions\RuntimeException;

use function Rubix\ML\argmax;

use function count;
use function array_unique;
use function array_merge;
use function array_pop;
use SplObjectStorage;

/**
 * Squared distance
 *
 * Euclidean distance without square root.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Sami Finnilä
 */
class SquaredDistance implements Distance
{
    /**
     * Return the data types that this kernel is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility(): array
    {
        return [
            DataType::continuous(),
        ];
    }

    /**
     * Compute the distance between two vectors.
     *
     * @internal
     *
     * @param list<int|float> $a
     * @param list<int|float> $b
     * @return float
     */
    public function compute(array $a, array $b): float
    {
        $distance = 0.0;

        foreach ($a as $i => $value) {
            $distance += ($value - $b[$i]) ** 2;
        }

        return $distance;
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString(): string
    {
        return 'Squared distance';
    }
}

/**
 * HDBSCAN
 *
 * *Hierarchical Density-Based Spatial Clustering of Applications with Noise* is a clustering algorithm
 * able to find non-linearly separable, arbitrarily-shaped clusters from a space with varying amounts of noise. 
 * The only mandatory parameters for the algorithm are a minimum cluster size and a smoothing
 * factor, sample size, that is used for estimating local probability density.  
 * HDBSCAN also has the ability to mark outliers as *noise*
 * and thus can be used as a *quasi* anomaly detector.
 *
 * References:  March W.B., Ram P., Gray A.G.
 *              Fast Euclidean Minimum Spanning Tree: Algorithm, Analysis, and Applications
 *              Proc. ACM SIGKDD’10, 2010, 603-611, https://mlpack.org/papers/emst.pdf
 * 
 *              Curtin, R., March, W., Ram, P., Anderson, D., Gray, A., & Isbell, C. (2013, May). 
 *              Tree-independent dual-tree algorithms. 
 *              In International Conference on Machine Learning (pp. 1435-1443). PMLR.
 *
 * @author      Sami Finnilä
 */
class HDBSCAN //implements Estimator

{
    /**
     * The minimum number of samples that can be considered to form a cluster.
     * Larger values will generate more stable clusters.
     *
     * @var int
     */
    protected int $minClusterSize;

    /**
     * The maximum length edge allowed within a cluster.
     *
     * @var float
     */
    protected float $maxEdgeLength;

    /**
     * The number of neighbors used for determining core distance when 
     * calculating mutual reachability distance between points.
     *
     * @var int
     */
    protected int $sampleSize;

    /**
     * The spatial tree used to run range searches.
     *
     * @var \Rubix\ML\Graph\Trees\Spatial
     */
    protected MstSolver $mstSolver;


    /**
     * The distance kernel used for computing interpoint distances.
     *
     * @var \Rubix\ML\Graph\Trees\Spatial
     */
    protected \Rubix\ML\Datasets\Labeled $dataset;



    /**
     * @param Labeled $dataset
     * @param array $oldCoreDistances
     * @param int $minClusterSize
     * @param int $sampleSize
     * @param float $maxEdgeLength
     * @param Distance $kernel
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(Labeled $dataset, int $minClusterSize = 5, int $sampleSize = 5, array $oldCoreDistances = [], ?Distance $kernel = null, bool $useTrueMst = true)
    {
        if ($minClusterSize < 2) {
            throw new InvalidArgumentException('Minimum cluster size must be'
                . " greater than 1, $minClusterSize.");
        }

        if ($sampleSize < 2) {
            throw new InvalidArgumentException('Minimum sample size must be'
                . " greater than 1, $sampleSize given.");
        }

        $kernel = $kernel ?? new SquaredDistance();
        $this->minClusterSize = $minClusterSize;
        $this->mstSolver = new MstSolver($dataset, 20, $sampleSize, $kernel, $oldCoreDistances, $useTrueMst);
    }

    public function getCoreNeighborDistances(): array
    {
        return $this->mstSolver->getCoreNeighborDistances();
    }

    /**
     * Return the estimator type.
     *
     * @return \Rubix\ML\EstimatorType
     */
    public function type(): EstimatorType
    {
        return EstimatorType::clusterer();
    }

    /**
     * Return the data types that the estimator is compatible with.
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility(): array
    {
        return $this->mstSolver->kernel()->compatibility();
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @return mixed[]
     */
    public function params(): array
    {
        return [
            'min cluster size' => $this->minClusterSize,
            'sample size' => $this->sampleSize,
            'dual tree' => $this->mstSolver,
        ];
    }

    /**
     * Form clusters and make predictions from the dataset (hard clustering).
     * 
     * @return list<MstClusterer>//@return list<int>
     */
    public function predict(): array
    {
        // Boruvka algorithm for MST generation        
        $edges = $this->mstSolver->getMst();

        // Boruvka complete, $edges now contains our mutual reachability distance MST
        if ($this->mstSolver->kernel() instanceof SquaredDistance) {
            foreach ($edges as &$edge) {
                $edge["distance"] = sqrt($edge["distance"]);
            }
        }
        unset($edge);

        // TODO: Min cluster separation/edge length of MstClusterer to the caller of this class
        $mstClusterer = new MstClusterer($edges, null, $this->minClusterSize, null, 0.0);
        $flatClusters = $mstClusterer->processCluster();

        return $flatClusters;
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString(): string
    {
        return 'HDBSCAN (' . Params::stringify($this->params()) . ')';
    }
}

class MstSolver
{
    private MrdBallTree $tree;
    private Distance $kernel;
    private int $radiusDiamFactor;
    private bool $useTrueMst;

    public function __construct(Labeled $fullDataset, int $maxLeafSize = 20, int $sampleSize = 5, ?Distance $kernel = null, ?array $oldCoreDistances = null, bool $useTrueMst = true)
    {
        $this->kernel = $kernel ?? new SquaredDistance();
        $this->radiusDiamFactor = $this->kernel instanceof SquaredDistance ? 4 : 2;

        $this->tree = new MrdBallTree($maxLeafSize, $sampleSize, $this->kernel);

        $this->tree->grow($fullDataset);
        $this->tree->precalculateCoreDistances($oldCoreDistances);

        $this->useTrueMst = $useTrueMst;
    }

    public function kernel(): Distance
    {
        return $this->kernel;
    }

    public function getCoreNeighborDistances(): array
    {
        return $this->tree->getCoreNeighborDistances();
    }

    public function getTree(): MrdBallTree
    {
        return $this->tree;
    }

    private function updateEdges($queryNode, $referenceNode, array &$newEdges, array &$vertexToSetId): void
    {

        $querySamples = $queryNode->dataset()->samples();
        $queryLabels = $queryNode->dataset()->labels();
        $referenceSamples = $referenceNode->dataset()->samples();
        $referenceLabels = $referenceNode->dataset()->labels();

        $longestDistance = 0.0;
        $shortestDistance = INF;

        foreach ($querySamples as $queryKey => $querySample) {
            $queryLabel = $queryLabels[$queryKey];
            $querySetId = $vertexToSetId[$queryLabel];

            if ($this->tree->getCoreDistance($queryLabel) > ($newEdges[$querySetId]["distance"] ?? INF)) {
                // The core distance of the current vertex is greater than the current best edge
                // for this setId. This means that the MRD will always be greater than the current best.
                continue;
            }

            foreach ($referenceSamples as $referenceKey => $referenceSample) {
                $referenceLabel = $referenceLabels[$referenceKey];
                $referenceSetId = $vertexToSetId[$referenceLabel];

                if ($querySetId === $referenceSetId) {
                    continue;
                }

                $distance = $this->tree->computeMrd($queryLabel, $querySample, $referenceLabel, $referenceSample);

                if ($distance < ($newEdges[$querySetId]["distance"] ?? INF)) {
                    $newEdges[$querySetId] = ["vertexFrom" => $queryLabel, "vertexTo" => $referenceLabel, "distance" => $distance];
                }
            }
            $candidateDist = $newEdges[$querySetId]["distance"] ?? INF;
            if ($candidateDist > $longestDistance) {
                $longestDistance = $candidateDist;
            }

            if ($candidateDist < $shortestDistance) {
                $shortestDistance = $candidateDist;
            }
        }

        // Update the bound of the query node        
        if ($this->kernel instanceof SquaredDistance) {
            $longestDistance = min($longestDistance, (2 * sqrt($queryNode->radius()) + sqrt($shortestDistance)) ** 2);
        } else {
            $longestDistance = min($longestDistance, 2 * $queryNode->radius() + $shortestDistance);
        }

        $queryNode->setLongestDistance($longestDistance);
    }

    private function findSetNeighbors($queryNode, $referenceNode, array &$newEdges, array &$vertexToSetId): void
    {
        if ($queryNode->isFullyConnected() && $referenceNode->isFullyConnected()) {
            if ($queryNode->getSetId() === $referenceNode->getSetId()) {
                // These nodes are connected and in the same set, so we can prune this reference node.
                return;
            }
        }

        // if d(Q,R) > d(Q) then
        //  return;

        $nodeDistance = $this->tree->nodeDistance($queryNode, $referenceNode);

        if ($nodeDistance > 0.0) {
            // Calculate smallest possible bound (i.e., d(Q) ):
            if ($queryNode->isFullyConnected()) {
                $currentBound = min($newEdges[$queryNode->getSetId()]["distance"] ?? INF, $queryNode->getLongestDistance());
            } else {
                $currentBound = $queryNode->getLongestDistance();
            }
            // If node distance is greater than the longest possible edge in this node,
            // prune this reference node
            if ($nodeDistance > $currentBound) {
                return;
            }
        }

        if ($queryNode instanceof DualTreeClique && $referenceNode instanceof DualTreeClique) {
            $this->updateEdges($queryNode, $referenceNode, $newEdges, $vertexToSetId);
            return;
        }

        if ($queryNode instanceof DualTreeClique) {
            foreach ($referenceNode->children() as $child) {
                $this->findSetNeighbors($queryNode, $child, $newEdges, $vertexToSetId);
            }
            return;
        }

        if ($referenceNode instanceof DualTreeClique) {
            $longestDistance = 0.0;

            $queryLeft = $queryNode->left();
            $queryRight = $queryNode->right();

            $this->findSetNeighbors($queryLeft, $referenceNode, $newEdges, $vertexToSetId);
            $this->findSetNeighbors($queryRight, $referenceNode, $newEdges, $vertexToSetId);

        } else { // if ($queryNode instanceof DualTreeBall && $referenceNode instanceof DualTreeBall)            
            $queryLeft = $queryNode->left();
            $queryRight = $queryNode->right();
            $referenceLeft = $referenceNode->left();
            $referenceRight = $referenceNode->right();

            $this->findSetNeighbors($queryLeft, $referenceLeft, $newEdges, $vertexToSetId);
            $this->findSetNeighbors($queryRight, $referenceRight, $newEdges, $vertexToSetId);
            $this->findSetNeighbors($queryLeft, $referenceRight, $newEdges, $vertexToSetId);
            $this->findSetNeighbors($queryRight, $referenceLeft, $newEdges, $vertexToSetId);
        }

        $longestLeft = $queryLeft->getLongestDistance();
        $longestRight = $queryRight->getLongestDistance();

        // TODO: min($longestLeft, $longestRight) + 2 * ($queryNode->radius()) <--- Can be made tighter?
        if ($this->kernel instanceof SquaredDistance) {
            $longestDistance = max($longestLeft, $longestRight);
            $longestLeft = (sqrt($longestLeft) + 2 * (sqrt($queryNode->radius()) - sqrt($queryLeft->radius()))) ** 2;
            $longestRight = (sqrt($longestRight) + 2 * (sqrt($queryNode->radius()) - sqrt($queryRight->radius()))) ** 2;
            $longestDistance = min($longestDistance, min($longestLeft, $longestRight), (sqrt(min($longestLeft, $longestRight)) + 2 * (sqrt($queryNode->radius()))) ** 2);
        } else {
            $longestDistance = max($longestLeft, $longestRight);
            $longestLeft = $longestLeft + 2 * ($queryNode->radius() - $queryLeft->radius());
            $longestRight = $longestRight + 2 * ($queryNode->radius() - $queryRight->radius());
            $longestDistance = min($longestDistance, min($longestLeft, $longestRight), min($longestLeft, $longestRight) + 2 * ($queryNode->radius()));
        }

        $queryNode->setLongestDistance($longestDistance);

        return;
    }

    public function getMst(): array
    {
        $edges = [];

        // MST generation using dual-tree boruvka algorithm

        $treeRoot = $this->tree->getRoot();

        $treeRoot->resetFullyConnectedStatus();

        $allLabels = $this->tree->getDataset()->labels();

        $vertexToSetId = array_combine($allLabels, range(0, count($allLabels) - 1));

        $vertexSets = [];
        foreach ($vertexToSetId as $vertex => $setId) {
            $vertexSets[$setId] = [$vertex];
        }

        if (!$this->useTrueMst) {
            $treeRoot->resetLongestEdge();
        }

        // Use nearest neighbors known from determining core distances for each vertex to
        // get the first set of $newEdges (we essentially can skip the first round of Boruvka):
        $newEdges = [];

        foreach ($allLabels as $label) {
            [$coreNeighborLabels, $coreNeighborDistances] = $this->tree->getCoreNeighbors($label);

            $coreDistance = end($coreNeighborDistances);

            foreach ($coreNeighborLabels as $neighborLabel) {
                if ($neighborLabel === $label) {
                    continue;
                }

                if ($this->tree->getCoreDistance($neighborLabel) <= $coreDistance) {
                    // This point is our nearest neighbor in mutual reachability terms, so
                    // an edge spanning these vertices will belong to the MST.
                    $newEdges[] = ["vertexFrom" => $label, "vertexTo" => $neighborLabel, "distance" => $coreDistance];
                    break;
                }
            }
        }

        ///////////////////////////////////////////////////////////////////////////////
        // Main dual tree Boruvka loop:

        while (true) {
            //Add new edges
            //Update vertex to set/set to vertex mappings
            foreach ($newEdges as $connectingEdge) {
                $setId1 = $vertexToSetId[$connectingEdge["vertexFrom"]];
                $setId2 = $vertexToSetId[$connectingEdge["vertexTo"]];

                if ($setId1 === $setId2) {
                    // These sets have already been merged earlier in this loop
                    continue;
                }

                $edges[] = $connectingEdge;

                if (count($vertexSets[$setId1]) < count($vertexSets[$setId2])) {
                    // Make a switch such that the larger set is always Id1
                    [$setId1, $setId2] = [$setId2, $setId1];
                }

                // Assign all vertices in set 2 to set 1
                foreach ($vertexSets[$setId2] as $vertexLabel) {
                    $vertexToSetId[$vertexLabel] = $setId1;
                }

                $vertexSets[$setId1] = array_merge($vertexSets[$setId1], $vertexSets[$setId2]);
                unset($vertexSets[$setId2]);

            }

            // Check for exit condition
            if (count($vertexSets) === 1) {
                break;
            }

            //Update the tree            
            if ($this->useTrueMst || empty($newEdges)) {
                $treeRoot->resetLongestEdge();
            }

            if (!empty($newEdges)) {
                $treeRoot->propagateSetChanges($vertexToSetId);
            }

            // Clear the array for a set of new edges
            $newEdges = [];

            $this->findSetNeighbors($treeRoot, $treeRoot, $newEdges, $vertexToSetId);

        }

        return $edges;
    }
}

class MrdBallTree extends BallTree
{

    private Labeled $dataset;
    private array $nativeInterpointCache;
    private array $coreDistances;
    private array $coreNeighborDistances;
    private int $sampleSize;
    private array $nodeDistances;
    private SplObjectStorage $nodeIds;
    private float $radiusDiamFactor;

    /**
     * @param int $maxLeafSize
     * @param int $coreDistSampleSize
     * @param \Rubix\ML\Kernels\Distance\Distance|null $kernel
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(int $maxLeafSize = 30, int $sampleSize = 5, ?Distance $kernel = null)
    {
        if ($maxLeafSize < 1) {
            throw new InvalidArgumentException('At least one sample is required'
                . " to form a leaf node, $maxLeafSize given.");
        }

        $this->maxLeafSize = $maxLeafSize;
        $this->sampleSize = $sampleSize;

        $this->kernel = $kernel ?? new SquaredDistance();
        $this->radiusDiamFactor = $this->kernel instanceof SquaredDistance ? 4 : 2;

        $this->nodeDistances = [];
        $this->nodeIds = new SplObjectStorage();
    }

    public function getCoreNeighborDistances(): array
    {
        return $this->coreNeighborDistances;
    }

    public function nodeDistance($queryNode, $referenceNode): float
    {
        // Use cache to accelerate repeated queries
        if ($this->nodeIds->contains($queryNode)) {
            $queryNodeId = $this->nodeIds[$queryNode];
        } else {
            $queryNodeId = $this->nodeIds->count();
            $this->nodeIds[$queryNode] = $queryNodeId;
        }

        if ($this->nodeIds->contains($referenceNode)) {
            $referenceNodeId = $this->nodeIds[$referenceNode];
        } else {
            $referenceNodeId = $this->nodeIds->count();
            $this->nodeIds[$referenceNode] = $referenceNodeId;
        }

        if ($referenceNodeId === $queryNodeId) {
            return (-$this->radiusDiamFactor * $queryNode->radius());
        }

        $smallIndex = min($queryNodeId, $referenceNodeId);
        $largeIndex = max($queryNodeId, $referenceNodeId);

        if (isset($this->nodeDistances[$smallIndex][$largeIndex])) {

            $nodeDistance = $this->nodeDistances[$smallIndex][$largeIndex];
        } else {

            $nodeDistance = $this->kernel->compute($queryNode->center(), $referenceNode->center());

            if ($this->kernel instanceof SquaredDistance) {
                $nodeDistance = sqrt($nodeDistance) - sqrt($queryNode->radius()) - sqrt($referenceNode->radius());
                $nodeDistance = abs($nodeDistance) * $nodeDistance;

            } else {
                $nodeDistance = $nodeDistance - $queryNode->radius() - $referenceNode->radius();

            }

            $this->nodeDistances[$smallIndex][$largeIndex] = $nodeDistance;
        }

        return $nodeDistance;

    }

    /**
     * Get tree root.
     * 
     * @internal
     * 
     * @return DualTreeBall 
     */
    public function getRoot(): Ball
    {
        return $this->root;
    }

    /**
     * Get the dataset the tree was grown on.
     * 
     * @internal
     * 
     * @return Labeled 
     */
    public function getDataset(): Labeled
    {
        return $this->dataset;
    }

    private function updateNearestNeighbors($queryNode, $referenceNode, $k, $maxRange, &$bestDistances): void
    {
        $querySamples = $queryNode->dataset()->samples();
        $queryLabels = $queryNode->dataset()->labels();
        $referenceSamples = $referenceNode->dataset()->samples();
        $referenceLabels = $referenceNode->dataset()->labels();

        $longestDistance = 0.0;
        $shortestDistance = INF;

        foreach ($querySamples as $queryKey => $querySample) {
            $queryLabel = $queryLabels[$queryKey];

            foreach ($referenceSamples as $referenceKey => $referenceSample) {
                $referenceLabel = $referenceLabels[$referenceKey];

                if ($queryLabel === $referenceLabel) {
                    continue;
                }

                // Calculate native distance
                $distance = $this->cachedComputeNative($queryLabel, $querySample, $referenceLabel, $referenceSample);

                if ($distance < $bestDistances[$queryLabel]) {
                    $this->coreNeighborDistances[$queryLabel][$referenceLabel] = $distance;

                    if (count($this->coreNeighborDistances[$queryLabel]) >= $k) {
                        asort($this->coreNeighborDistances[$queryLabel]);
                        $this->coreNeighborDistances[$queryLabel] = array_slice($this->coreNeighborDistances[$queryLabel], 0, $k, true);
                        $bestDistances[$queryLabel] = min(end($this->coreNeighborDistances[$queryLabel]), $maxRange);
                    }
                }
            }

            if ($bestDistances[$queryLabel] > $longestDistance) {
                $longestDistance = $bestDistances[$queryLabel];
            }

            if ($bestDistances[$queryLabel] < $shortestDistance) {
                $shortestDistance = $bestDistances[$queryLabel];
            }
        }

        if ($this->kernel instanceof SquaredDistance) {
            $longestDistance = min($longestDistance, (2 * sqrt($queryNode->radius()) + sqrt($shortestDistance)) ** 2);
        } else {
            $longestDistance = min($longestDistance, 2 * $queryNode->radius() + $shortestDistance);
        }
        $queryNode->setLongestDistance($longestDistance);
    }

    private function findNearestNeighbors($queryNode, $referenceNode, $k, $maxRange, &$bestDistances): void
    {
        $nodeDistance = $this->nodeDistance($queryNode, $referenceNode);

        if ($nodeDistance > 0.0) {
            // Calculate smallest possible bound (i.e., d(Q) ):
            $currentBound = $queryNode->getLongestDistance();

            // If node distance is greater than the longest possible edge in this node,
            // prune this reference node
            if ($nodeDistance > $currentBound) {
                return;
            }
        }

        if ($queryNode instanceof DualTreeClique && $referenceNode instanceof DualTreeClique) {
            $this->updateNearestNeighbors($queryNode, $referenceNode, $k, $maxRange, $bestDistances);
            return;
        }

        if ($queryNode instanceof DualTreeClique) {
            foreach ($referenceNode->children() as $child) {
                $this->findNearestNeighbors($queryNode, $child, $k, $maxRange, $bestDistances);
            }
            return;
        }

        if ($referenceNode instanceof DualTreeClique) {
            $longestDistance = 0.0;

            $queryLeft = $queryNode->left();
            $queryRight = $queryNode->right();

            $this->findNearestNeighbors($queryLeft, $referenceNode, $k, $maxRange, $bestDistances);
            $this->findNearestNeighbors($queryRight, $referenceNode, $k, $maxRange, $bestDistances);

        } else {

            // --> if ($queryNode instanceof DualTreeBall && $referenceNode instanceof DualTreeBall)
            $queryLeft = $queryNode->left();
            $queryRight = $queryNode->right();
            $referenceLeft = $referenceNode->left();
            $referenceRight = $referenceNode->right();

            // TODO: traverse closest neighbor nodes first
            $this->findNearestNeighbors($queryLeft, $referenceLeft, $k, $maxRange, $bestDistances);
            $this->findNearestNeighbors($queryRight, $referenceRight, $k, $maxRange, $bestDistances);
            $this->findNearestNeighbors($queryLeft, $referenceRight, $k, $maxRange, $bestDistances);
            $this->findNearestNeighbors($queryRight, $referenceLeft, $k, $maxRange, $bestDistances);
        }

        $longestLeft = $queryLeft->getLongestDistance();
        $longestRight = $queryRight->getLongestDistance();

        // TODO: min($longestLeft, $longestRight) + 2 * ($queryNode->radius()) <--- Can be made tighter by using the shortest distance from child.
        if ($this->kernel instanceof SquaredDistance) {
            $longestDistance = max($longestLeft, $longestRight);
            $longestLeft = (sqrt($longestLeft) + 2 * (sqrt($queryNode->radius()) - sqrt($queryLeft->radius()))) ** 2;
            $longestRight = (sqrt($longestRight) + 2 * (sqrt($queryNode->radius()) - sqrt($queryRight->radius()))) ** 2;
            $longestDistance = min($longestDistance, min($longestLeft, $longestRight), (sqrt(min($longestLeft, $longestRight)) + 2 * (sqrt($queryNode->radius()))) ** 2);
        } else {
            $longestDistance = max($longestLeft, $longestRight);
            $longestLeft = $longestLeft + 2 * ($queryNode->radius() - $queryLeft->radius());
            $longestRight = $longestRight + 2 * ($queryNode->radius() - $queryRight->radius());
            $longestDistance = min($longestDistance, min($longestLeft, $longestRight), min($longestLeft, $longestRight) + 2 * ($queryNode->radius()));
        }

        $queryNode->setLongestDistance($longestDistance);

        return;
    }

    public function kNearestAll($k, float $maxRange = INF): void
    {
        $this->coreNeighborDistances = [];

        $allLabels = $this->dataset->labels();
        $bestDistances = [];
        foreach ($allLabels as $label) {
            $bestDistances[$label] = $maxRange;
        }

        $treeRoot = $this->root;

        $treeRoot->resetFullyConnectedStatus();
        $treeRoot->resetLongestEdge();

        $this->findNearestNeighbors($treeRoot, $treeRoot, $k, $maxRange, $bestDistances);
    }

    /**
     * Precompute core distances for the current dataset to accelerate
     * subsequent MRD queries. Optionally, utilize core distances that've
     * been previously determined for (a subset of) the current dataset.
     * Returns the updated core distances for future use.
     *
     * @internal
     *
     * @param array|null $oldCoreNeighbors
     * @return array
     */

    public function precalculateCoreDistances(?array $oldCoreNeighbors = null)
    {
        if (empty($this->dataset)) {
            throw new \Exception("Precalculation of core distances requested but dataset is empty. Call ->grow() first!");
        }

        $labels = $this->dataset->labels();

        if ($oldCoreNeighbors !== null && !empty($oldCoreNeighbors) && count(reset($oldCoreNeighbors)) >= $this->sampleSize) {

            // Determine the search radius for core distances based on the largest old
            // core distance (points father than that cannot shorten the old core distances)
            $largestOldCoreDistance = 0.0;

            // Utilize old (possibly stale) core distance data
            foreach ($oldCoreNeighbors as $label => $oldDistances) {
                $coreDistance = (array_values($oldDistances))[$this->sampleSize - 1];

                if ($coreDistance > $largestOldCoreDistance) {
                    $largestOldCoreDistance = $coreDistance;
                }

                $this->coreNeighborDistances[$label] = $oldDistances;
                $this->coreDistances[$label] = $coreDistance;
            }

            $updatedOldCoreLabels = [];

            // Don't recalculate core distances for the old labels
            $labels = array_filter($labels, function ($label) use ($oldCoreNeighbors) {
                return !isset($oldCoreNeighbors[$label]);
            });

            foreach ($labels as $label) {
                [$neighborLabels, $neighborDistances] = $this->cachedRange($label, $largestOldCoreDistance);
                // TODO: cachedRange may not return $this->sampleSize number of labels.
                $this->coreNeighborDistances[$label] = array_combine(array_slice($neighborLabels, 0, $this->sampleSize), array_slice($neighborDistances, 0, $this->sampleSize));
                $this->coreDistances[$label] = $neighborDistances[$this->sampleSize - 1];

                // If one of the old vertices is within the update radius of this new vertex,
                // check whether the old core distance needs to be updated.
                foreach ($neighborLabels as $distanceKey => $neighborLabel) {
                    if (isset($oldCoreNeighbors[$neighborLabel])) {
                        $newDistance = $neighborDistances[$distanceKey];
                        if ($newDistance < $this->coreDistances[$neighborLabel]) {
                            $this->coreNeighborDistances[$neighborLabel][$label] = $newDistance;
                            $updatedOldCoreLabels[$neighborLabel] = true;
                        }

                    }
                }
            }

            foreach (array_keys($updatedOldCoreLabels) as $label) {
                asort($this->coreNeighborDistances[$label]);
                $this->coreNeighborDistances[$label] = array_slice($this->coreNeighborDistances[$label], 0, $this->sampleSize, true);
                $this->coreDistances[$label] = end($this->coreNeighborDistances[$label]);
            }
        } else { // $oldCoreNeighbors === null
            $this->kNearestAll($this->sampleSize, INF);

            foreach ($this->dataset->labels() as $label) {
                $this->coreDistances[$label] = end($this->coreNeighborDistances[$label]);
            }
        }

        return $this->coreNeighborDistances;
    }

    /**
     * Inserts a new neighbor to core neighbors if the distance
     * is greater than the current largest distance for the query label.
     * 
     * Returns the updated core distance or INF if there are less than $this->sampleSize neighbors.
     * 
     * @internal
     *
     * @param mixed $queryLabel
     * @param mixed $referenceLabel
     * @param float $distance
     * @return float
     */
    private function insertToCoreDistances($queryLabel, $referenceLabel, $distance): float
    {
        // Update the core distances of the queryLabel
        if (isset($this->coreDistances[$queryLabel])) {
            if ($this->coreDistances[$queryLabel] > $distance) {
                $this->coreNeighborDistances[$queryLabel][$referenceLabel] = $distance;
                asort($this->coreNeighborDistances[$queryLabel]);

                $this->coreNeighborDistances[$queryLabel] = array_slice($this->coreNeighborDistances[$queryLabel], 0, $this->sampleSize, true);
                $this->coreDistances[$queryLabel] = end($this->coreNeighborDistances[$queryLabel]);
            }
        } else {
            $this->coreNeighborDistances[$queryLabel][$referenceLabel] = $distance;

            if (count($this->coreNeighborDistances[$queryLabel]) >= $this->sampleSize) {
                asort($this->coreNeighborDistances[$queryLabel]);

                $this->coreNeighborDistances[$queryLabel] = array_slice($this->coreNeighborDistances[$queryLabel], 0, $this->sampleSize, true);
                $this->coreDistances[$queryLabel] = end($this->coreNeighborDistances[$queryLabel]);
            }
        }

        // Update the core distances of the referenceLabel (this is not necessary, but *may* accelerate the algo slightly)
        if (isset($this->coreDistances[$referenceLabel])) {
            if ($this->coreDistances[$referenceLabel] > $distance) {
                $this->coreNeighborDistances[$referenceLabel][$queryLabel] = $distance;
                asort($this->coreNeighborDistances[$referenceLabel]);

                $this->coreNeighborDistances[$referenceLabel] = array_slice($this->coreNeighborDistances[$referenceLabel], 0, $this->sampleSize, true);
                $this->coreDistances[$referenceLabel] = end($this->coreNeighborDistances[$referenceLabel]);
            }
        } else {
            $this->coreNeighborDistances[$referenceLabel][$queryLabel] = $distance;

            if (count($this->coreNeighborDistances[$referenceLabel]) > $this->sampleSize) {
                asort($this->coreNeighborDistances[$referenceLabel]);
                $this->coreNeighborDistances[$referenceLabel] = array_slice($this->coreNeighborDistances[$referenceLabel], 0, $this->sampleSize, true);
                $this->coreDistances[$referenceLabel] = end($this->coreNeighborDistances[$referenceLabel]);
            }
        }

        return $this->coreDistances[$queryLabel] ?? INF;
    }

    /**
     * Compute the mutual reachability distance between two vectors.
     *
     * @internal
     *
     * @param int|string $a 
     * @param int|string $b 
     * @return float
     */
    public function computeMrd($a, array $a_vector, $b, array $b_vector): float
    {
        $distance = $this->cachedComputeNative($a, $a_vector, $b, $b_vector);

        return max($distance, $this->getCoreDistance($a), $this->getCoreDistance($b));
    }

    public function getCoreDistance($label): float
    {
        if (!isset($this->coreDistances[$label])) {
            [$labels, $distances] = $this->getCoreNeighbors($label);
            $this->coreDistances[$label] = end($distances);
        }

        return $this->coreDistances[$label];
    }

    public function cachedComputeNative($a, array $a_vector, $b, array $b_vector, bool $storeNewCalculations = true): float
    {
        if (isset($this->coreNeighborDistances[$a][$b])) {
            return $this->coreNeighborDistances[$a][$b];
        }
        if (isset($this->coreNeighborDistances[$b][$a])) {
            return $this->coreNeighborDistances[$b][$a];
        }

        if ($a < $b) {
            $smallIndex = $a;
            $largeIndex = $b;
        } else {
            $smallIndex = $b;
            $largeIndex = $a;
        }

        if (!isset($this->nativeInterpointCache[$smallIndex][$largeIndex])) {
            $distance = $this->kernel->compute($a_vector, $b_vector);
            if ($storeNewCalculations) {
                $this->nativeInterpointCache[$smallIndex][$largeIndex] = $distance;
            }
            return $distance;
        }

        return $this->nativeInterpointCache[$smallIndex][$largeIndex];
    }

    /**
     * Run a n nearest neighbors search on a single label and return the neighbor labels, and distances in a 2-tuple
     * 
     *
     * @internal
     *
     * @param int|string $sampleLabel
     * @param bool $useCachedValues
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return array{list<mixed>,list<float>}
     */
    public function getCoreNeighbors($sampleLabel, bool $useCachedValues = true): array
    {
        if ($useCachedValues && isset($this->coreNeighborDistances[$sampleLabel])) {
            return [array_keys($this->coreNeighborDistances[$sampleLabel]), array_values($this->coreNeighborDistances[$sampleLabel])];
        }

        $sampleKey = array_search($sampleLabel, $this->dataset->labels());
        $sample = $this->dataset->sample($sampleKey);

        $squaredDistance = $this->kernel instanceof SquaredDistance;

        /** @var list<DualTreeBall|DualTreeClique> **/
        $stack = [$this->root];
        $stackDistances = [0.0];
        $radius = INF;

        $labels = $distances = [];

        while ($current = array_pop($stack)) {
            $currentDistance = array_pop($stackDistances);

            if ($currentDistance > $radius) {
                continue;
            }

            if ($current instanceof DualTreeBall) {
                foreach ($current->children() as $child) {
                    if ($child instanceof Hypersphere) {
                        $distance = $this->kernel->compute($sample, $child->center());

                        if ($squaredDistance) {
                            $distance = sqrt($distance);
                            $childRadius = sqrt($child->radius());
                            $distance = $distance - $childRadius;
                            $distance = abs($distance) * $distance;
                        } else {
                            $distance = $distance - $child->radius();
                        }

                        if ($distance < $radius) {
                            $stack[] = $child;
                            $stackDistances[] = $distance;
                        }
                    }
                }
                array_multisort($stackDistances, SORT_DESC, $stack);

            } elseif ($current instanceof DualTreeClique) {
                $dataset = $current->dataset();
                $neighborLabels = $dataset->labels();

                foreach ($dataset->samples() as $i => $neighbor) {
                    if ($neighborLabels[$i] === $sampleLabel) {
                        continue;
                    }

                    $distance = $this->cachedComputeNative($sampleLabel, $sample, $neighborLabels[$i], $neighbor);

                    if ($distance <= $radius) {
                        $labels[] = $neighborLabels[$i];
                        $distances[] = $distance;
                    }
                }

                if (count($labels) >= $this->sampleSize) {
                    array_multisort($distances, $labels);
                    $radius = $distances[$this->sampleSize - 1];
                    $labels = array_slice($labels, 0, $this->sampleSize);
                    $distances = array_slice($distances, 0, $this->sampleSize);
                }
            }
        }
        return [$labels, $distances];
    }

    /**
     * Return all labels, and distances within a given radius of a sample.
     * 
     *
     * @internal
     *
     * @param int $sampleLabel
     * @param float $radius
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return array{list<mixed>,list<float>}
     */
    public function cachedRange($sampleLabel, float $radius): array
    {
        $sampleKey = array_search($sampleLabel, $this->dataset->labels());
        $sample = $this->dataset->sample($sampleKey);

        $squaredDistance = $this->kernel instanceof SquaredDistance;

        /** @var list<DualTreeBall|DualTreeClique> **/
        $stack = [$this->root];

        $labels = $distances = [];

        while ($current = array_pop($stack)) {

            if ($current instanceof DualTreeBall) {
                foreach ($current->children() as $child) {
                    if ($child instanceof Hypersphere) {
                        $distance = $this->kernel->compute($sample, $child->center());

                        if ($squaredDistance) {
                            $distance = sqrt($distance);
                            $childRadius = sqrt($child->radius());
                            $minDistance = $distance - $childRadius;
                            $minDistance = abs($minDistance) * $minDistance;
                            $maxDistance = ($distance + $childRadius) ** 2;
                        } else {
                            $childRadius = $child->radius();
                            $minDistance = $distance - $childRadius;
                            $maxDistance = $distance + $childRadius;
                        }

                        if ($minDistance < $radius) {

                            if ($maxDistance < $radius && $child instanceof DualTreeBall) {
                                // The whole child is within the specified radius: greedily add all sub-children recursively to the stack
                                $subStack = [$child];
                                while ($subStackCurrent = array_pop($subStack)) {
                                    foreach ($subStackCurrent->children() as $subChild) {
                                        if ($subChild instanceof DualTreeClique) {
                                            $stack[] = $subChild;
                                        } else {
                                            $subStack[] = $subChild;
                                        }
                                    }
                                }
                            } else {
                                $stack[] = $child;
                            }

                        }
                    }
                }

            } elseif ($current instanceof DualTreeClique) {
                $dataset = $current->dataset();
                $neighborLabels = $dataset->labels();

                foreach ($dataset->samples() as $i => $neighbor) {
                    $distance = $this->cachedComputeNative($sampleLabel, $sample, $neighborLabels[$i], $neighbor);

                    if ($distance <= $radius) {
                        $labels[] = $neighborLabels[$i];
                        $distances[] = $distance;
                    }
                }
            }
        }
        array_multisort($distances, $labels);
        return [$labels, $distances];
    }

    /**
     * Insert a root node and recursively split the dataset until a terminating
     * condition is met. This also sets the dataset that will be used to calculate
     * core distances. Previously calculated core distances will be stored/used
     * despite calling grow, unless precalculateCoreDistances() is called again.
     *
     * @internal
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function grow(Labeled $dataset): void
    {
        $this->dataset = $dataset;
        $this->root = DualTreeBall::split($dataset, $this->kernel);

        $stack = [$this->root];

        while ($current = array_pop($stack)) {
            [$left, $right] = $current->subsets();

            $current->cleanup();

            if ($left->numSamples() > $this->maxLeafSize) {
                $node = DualTreeBall::split($left, $this->kernel);

                $current->attachLeft($node);

                $stack[] = $node;
            } elseif (!$left->empty()) {
                $current->attachLeft(DualTreeClique::terminate($left, $this->kernel));
            }

            if ($right->numSamples() > $this->maxLeafSize) {
                $node = DualTreeBall::split($right, $this->kernel);

                if ($node->isPoint()) {
                    $current->attachRight(DualTreeClique::terminate($right, $this->kernel));
                } else {
                    $current->attachRight($node);

                    $stack[] = $node;
                }
            } elseif (!$right->empty()) {
                $current->attachRight(DualTreeClique::terminate($right, $this->kernel));
            }
        }
    }

}

class DualTreeBall extends Ball
{
    protected float $longestDistanceInNode;
    protected bool $fullyConnected;
    protected null|int|string $setId;


    public function setLongestDistance($longestDistance): void
    {
        $this->longestDistanceInNode = $longestDistance;
    }

    public function getLongestDistance(): float
    {
        return $this->longestDistanceInNode;
    }

    public function resetLongestEdge(): void
    {
        $this->longestDistanceInNode = INF;
        foreach ($this->children() as $child) {
            $child->resetLongestEdge();
        }
    }

    public function resetFullyConnectedStatus(): void
    {
        $this->fullyConnected = false;
        foreach ($this->children() as $child) {
            $child->resetFullyConnectedStatus();
        }
    }

    public function getSetId(): null|string|int
    {
        if (!$this->fullyConnected) {
            return null;
        }
        return $this->setId;
    }

    public function isFullyConnected(): bool
    {
        return $this->fullyConnected;
    }

    public function propagateSetChanges(array &$labelToSetId): null|int|string
    {
        if ($this->fullyConnected) {

            // If we are already fully connected, we just need to check if the
            // set id has changed
            foreach ($this->children() as $child) {
                $this->setId = $child->propagateSetChanges($labelToSetId);
            }

            return $this->setId;
        }

        // If, and only if, both children are fully connected and in the same set id then
        // we, too, are fully connected
        $setId = null;
        foreach ($this->children() as $child) {
            $retVal = $child->propagateSetChanges($labelToSetId);

            if ($retVal === null) {
                return null;
            }

            if ($setId !== null && $setId !== $retVal) {
                return null;
            }

            $setId = $retVal;
        }

        $this->setId = $setId;
        $this->fullyConnected = true;

        return $this->setId;
    }

    /**
     * Factory method to build a hypersphere by splitting the dataset into left and right clusters.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @param \Rubix\ML\Kernels\Distance\Distance $kernel
     * @return self
     */
    public static function split(Labeled $dataset, Distance $kernel): self
    {
        $center = [];

        foreach ($dataset->features() as $column => $values) {
            if ($dataset->featureType($column)->isContinuous()) {
                $center[] = Stats::mean($values);
            } else {
                $center[] = argmax(array_count_values($values));
            }
        }

        $distances = [];

        foreach ($dataset->samples() as $sample) {
            $distances[] = $kernel->compute($sample, $center);
        }

        $radius = max($distances) ?: 0.0;

        $leftCentroid = $dataset->sample(argmax($distances));

        $distances = [];

        foreach ($dataset->samples() as $sample) {
            $distances[] = $kernel->compute($sample, $leftCentroid);
        }

        $rightCentroid = $dataset->sample(argmax($distances));

        $subsets = $dataset->spatialSplit($leftCentroid, $rightCentroid, $kernel);

        return new self($center, $radius, $subsets);
    }
}

class DualTreeClique extends Clique
{
    protected float $longestDistanceInNode;
    protected bool $fullyConnected;
    protected null|int|string $setId;

    public function setLongestDistance($longestDistance): void
    {
        $this->longestDistanceInNode = $longestDistance;
    }

    public function getLongestDistance(): float
    {
        return $this->longestDistanceInNode;
    }

    public function resetLongestEdge(): void
    {
        $this->longestDistanceInNode = INF;
    }

    public function resetFullyConnectedStatus(): void
    {
        $this->fullyConnected = false;
    }

    public function getSetId(): null|string|int
    {
        if (!$this->fullyConnected) {
            return null;
        }

        return $this->setId;
    }

    public function isFullyConnected(): bool
    {
        return $this->fullyConnected;
    }

    public function propagateSetChanges(array &$labelToSetId): null|int|string
    {
        if ($this->fullyConnected) {
            $this->setId = $labelToSetId[$this->dataset->label(0)];
            return $this->setId;
        }

        $labels = $this->dataset->labels();

        $label =

            $setId = $labelToSetId[array_pop($labels)];

        foreach ($labels as $label) {
            if ($setId !== $labelToSetId[$label]) {
                return null;
            }
        }

        $this->fullyConnected = true;
        $this->setId = $setId;

        return $this->setId;
    }

    /**
     * Terminate a branch with a dataset.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @param \Rubix\ML\Kernels\Distance\Distance $kernel
     * @return self
     */
    public static function terminate(Labeled $dataset, Distance $kernel): self
    {
        $center = [];

        foreach ($dataset->features() as $column => $values) {
            if ($dataset->featureType($column)->isContinuous()) {
                $center[] = Stats::mean($values);
            } else {
                $center[] = argmax(array_count_values($values));
            }
        }

        $distances = [];

        foreach ($dataset->samples() as $sample) {
            $distances[] = $kernel->compute($sample, $center);
        }
        $radius = max($distances) ?: 0.0;
        return new self($dataset, $center, $radius);
    }
}


// TODO: core edges are not always stored properly (if two halves of the remaining clusters are both pruned at the same time)
// TODO: store vertex lambda length (relative to cluster lambda length) for all vertices for soft clustering.
class MstClusterer
{
    private array $edges;
    private array $remainingEdges;
    private float $startingLambda;
    private float $finalLambda;
    private float $clusterWeight;
    private int $minimumClusterSize;
    private array $coreEdges;
    private bool $isRoot;
    private array $mapVerticesToEdges;
    private float $minClusterSeparation;

    public function __construct(array $edges, ?array $mapVerticesToEdges, int $minimumClusterSize, ?float $startingLambda = null, float $minClusterSeparation = 0.1)
    {
        //Ascending sort of edges while perserving original keys.
        $this->edges = $edges;

        uasort($this->edges, function ($a, $b) {
            if ($a["distance"] > $b["distance"]) {
                return 1;
            }
            if ($a["distance"] < $b["distance"]) {
                return -1;
            }
            return 0;
        });

        $this->remainingEdges = $this->edges;

        if ($mapVerticesToEdges === null) {
            $mapVerticesToEdges = [];
            foreach ($this->edges as $edgeIndex => $edge) {
                $mapVerticesToEdges[$edge['vertexFrom']][$edgeIndex] = true;
                $mapVerticesToEdges[$edge['vertexTo']][$edgeIndex] = true;
            }
        }

        $this->mapVerticesToEdges = $mapVerticesToEdges;

        if (is_null($startingLambda)) {
            $this->isRoot = true;
            $this->startingLambda = 0.0;
        } else {
            $this->isRoot = false;
            $this->startingLambda = $startingLambda;
        }

        $this->minimumClusterSize = $minimumClusterSize;

        $this->coreEdges = [];

        $this->clusterWeight = 0.0;


        $this->minClusterSeparation = $minClusterSeparation;
    }

    public function processCluster(): array
    {
        $currentLambda = $lastLambda = $this->startingLambda;
        $edgeLength = INF;

        while (true) {
            $edgeCount = count($this->remainingEdges);

            if ($edgeCount < ($this->minimumClusterSize - 1)) {
                $this->finalLambda = $currentLambda;
                $this->coreEdges = $this->remainingEdges;

                return [$this];
            }

            $currentLongestEdgeKey = array_key_last($this->remainingEdges);
            $currentLongestEdge = array_pop($this->remainingEdges);

            $vertexConnectedFrom = $currentLongestEdge["vertexFrom"];
            $vertexConnectedTo = $currentLongestEdge["vertexTo"];
            $edgeLength = $currentLongestEdge["distance"];

            unset($this->mapVerticesToEdges[$vertexConnectedFrom][$currentLongestEdgeKey]);
            unset($this->mapVerticesToEdges[$vertexConnectedTo][$currentLongestEdgeKey]);

            if ($edgeLength > 0.0) {
                $currentLambda = 1 / $edgeLength;
            }

            $this->clusterWeight += ($currentLambda - $lastLambda) * $edgeCount;
            $lastLambda = $currentLambda;

            if (!$this->pruneFromCluster($vertexConnectedTo) && !$this->pruneFromCluster($vertexConnectedFrom)) {

                // This cluster will (probably) split into two child clusters:

                [$childClusterEdges1, $childClusterVerticesToEdges1] = $this->getChildClusterComponents($vertexConnectedTo);
                [$childClusterEdges2, $childClusterVerticesToEdges2] = $this->getChildClusterComponents($vertexConnectedFrom);

                if ($edgeLength < $this->minClusterSeparation) {

                    if (count($childClusterEdges1) > count($childClusterEdges2)) {
                        $this->remainingEdges = $childClusterEdges1;
                        $this->mapVerticesToEdges = $childClusterVerticesToEdges1;
                    } else {
                        $this->remainingEdges = $childClusterEdges2;
                        $this->mapVerticesToEdges = $childClusterVerticesToEdges2;
                    }
                    continue;
                }

                // Choose clusters using excess of mass method:
                // Return a list of children if the weight of all children is more than $this->clusterWeight.
                // Otherwise return the current cluster and discard the children. This way we "choose" a combination
                // of clusters that weigh the most (i.e. have most (excess of) mass). Always discard the root cluster.
                $this->finalLambda = $currentLambda;

                $childCluster1 = new MstClusterer($childClusterEdges1, $childClusterVerticesToEdges1, $this->minimumClusterSize, $this->finalLambda, $this->minClusterSeparation);
                $childCluster2 = new MstClusterer($childClusterEdges2, $childClusterVerticesToEdges2, $this->minimumClusterSize, $this->finalLambda, $this->minClusterSeparation);

                // Resolve all chosen child clusters recursively
                $childClusters = array_merge($childCluster1->processCluster(), $childCluster2->processCluster());

                $childrenWeight = 0.0;
                foreach ($childClusters as $childCluster) {
                    $childrenWeight += $childCluster->getClusterWeight();
                    array_merge($this->coreEdges, $childCluster->getCoreEdges());
                }

                if (($childrenWeight > $this->clusterWeight) || $this->isRoot) {
                    return $childClusters;
                }

                return [$this];
            }
        }
    }

    private function pruneFromCluster(int $vertexId): bool
    {
        $edgeIndicesToPrune = [];
        $verticesToPrune = [];
        $vertexStack = [$vertexId];

        while (!empty($vertexStack)) {
            $currentVertex = array_pop($vertexStack);
            $verticesToPrune[] = $currentVertex;

            if (count($verticesToPrune) >= $this->minimumClusterSize) {
                return false;
            }

            foreach (array_keys($this->mapVerticesToEdges[$currentVertex]) as $edgeKey) {
                if (isset($edgeIndicesToPrune[$edgeKey])) {
                    continue;
                }

                if ($this->remainingEdges[$edgeKey]["vertexFrom"] === $currentVertex) {
                    $vertexStack[] = $this->remainingEdges[$edgeKey]["vertexTo"];
                    $edgeIndicesToPrune[$edgeKey] = true;
                } elseif ($this->remainingEdges[$edgeKey]["vertexTo"] === $currentVertex) {
                    $vertexStack[] = $this->remainingEdges[$edgeKey]["vertexFrom"];
                    $edgeIndicesToPrune[$edgeKey] = true;
                }
            }
        }

        // Prune edges
        foreach (array_keys($edgeIndicesToPrune) as $edgeToPrune) {
            unset($this->remainingEdges[$edgeToPrune]);
        }

        // Prune vertices to edges map (not stricly necessary but saves some memory)
        foreach ($verticesToPrune as $vertexLabel) {
            unset($this->mapVerticesToEdges[$vertexLabel]);
        }

        return true;
    }

    private function getChildClusterComponents(int $vertexId): array
    {

        $vertexStack = [$vertexId];
        $edgeIndicesInCluster = [];
        $verticesInCluster = [];

        while (!empty($vertexStack)) {

            $currentVertex = array_pop($vertexStack);
            $verticesInCluster[$currentVertex] = $this->mapVerticesToEdges[$currentVertex];

            foreach (array_keys($this->mapVerticesToEdges[$currentVertex]) as $edgeKey) {

                if (isset($edgeIndicesInCluster[$edgeKey])) {
                    continue;
                }

                if ($this->remainingEdges[$edgeKey]["vertexFrom"] === $currentVertex) {
                    $vertexStack[] = $this->remainingEdges[$edgeKey]["vertexTo"];
                    $edgeIndicesInCluster[$edgeKey] = true;
                } elseif ($this->remainingEdges[$edgeKey]["vertexTo"] === $currentVertex) {
                    $vertexStack[] = $this->remainingEdges[$edgeKey]["vertexFrom"];
                    $edgeIndicesInCluster[$edgeKey] = true;
                }
            }
        }

        // Collecting the edges is done in a separate loop to perserve the ordering according to length.
        // (See constructor.)
        $edgesInCluster = [];
        foreach ($this->remainingEdges as $edgeKey => $edge) {
            if (isset($edgeIndicesInCluster[$edgeKey])) {
                $edgesInCluster[$edgeKey] = $edge;
            }
        }

        return [$edgesInCluster, $verticesInCluster];

    }

    public function getClusterWeight(): float
    {
        return $this->clusterWeight;
    }

    public function getVertexKeys(): array
    {
        $vertexKeys = [];

        foreach ($this->edges as $edge) {
            $vertexKeys[] = $edge["vertexTo"];
            $vertexKeys[] = $edge["vertexFrom"];
        }

        return array_unique($vertexKeys);
    }

    public function getCoreEdges(): array
    {
        return $this->coreEdges;
    }
}
