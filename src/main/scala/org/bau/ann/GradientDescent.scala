package org.bau.ann
/**
  * Created by foreks on 22.04.2017.
  */
trait GradientDescent extends Optimizer with MachineLearningModel {
  val nHidden: Int
  val weights: Array[Double]
  val learningRate = 0.1

  def fit(trainingData: Iterable[(IndexedSeq[Double], IndexedSeq[Double])]): Unit  = {
    for(pair <- trainingData){
      val actual = this(pair._1)
      val X = getH(pair)
      val desired = pair._2
      for( outputIndex <- desired.indices) {
        val error = desired(outputIndex) - actual(outputIndex)
        val update = X.indices.map(i => learningRate * X(i) * error)
        X.indices.foreach { i =>
          val weightIndex = (nHidden  * outputIndex) + i
          weights(weightIndex) = update(i) + weights(weightIndex)
        }
        //bias term
        val weightIndex = (nHidden  * outputIndex) + X.length
        weights(weightIndex) = learningRate * error + weights(weightIndex)
      }
    }
  }
}
