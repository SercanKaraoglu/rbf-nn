package org.bau

/**
  * Created by foreks on 22.04.2017.
  */
package object ann {
  type Features = IndexedSeq[Double]
  type ErrorFunc = (Function[IndexedSeq[Double], IndexedSeq[Double]], Iterable[(IndexedSeq[Double], IndexedSeq[Double])]) => Double

  type DistanceFunc = ((IndexedSeq[Double], IndexedSeq[Double]) => Double)
  type RadialBasisFunction = (IndexedSeq[Double] => Double)
  type Clustering = () => IndexedSeq[(Features, IndexedSeq[Features])]
  trait Optimizer {
    def fit(dataSet: Iterable[(IndexedSeq[Double], IndexedSeq[Double])])
  }
  trait MachineLearningModel extends (IndexedSeq[Double] => IndexedSeq[Double]) {
    def getH(input: (IndexedSeq[Double], IndexedSeq[Double])):IndexedSeq[Double];
  }
}
