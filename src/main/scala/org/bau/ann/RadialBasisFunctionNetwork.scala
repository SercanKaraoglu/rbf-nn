package org.bau.ann

/**
  * Created by sercan on 18.04.2017.
  */
object RadialBasisFunctionNetwork extends App {

  import java.io.File

  val numberPattern = "([-+]?[0-9]*\\.?[0-9]+[eE][-+]?[0-9]+)?".r
  val dataPath = new File("/home/foreks/git/rbf-nn/data")

  def getData(path: String) =
    scala.io.Source.fromFile(new File(dataPath, path)).getLines()
      .map { line =>
        val columns = numberPattern.findAllIn(line).filter(_ != "").map(_.toDouble).toVector
        columns.splitAt(columns.length - 1)
      }
      .toVector

  val trainingData = getData("d_reg_tra.txt")
  val testData = getData("d_reg_val.txt")


  case class KMeansImpl(override val seed: Int, override val k: Int, override val eta: Double, override val data: Vector[Features])
    extends KMeans(seed, k, eta, data)
      with EuclideanDistance

  def trainAndGetModel(rbfCount: Int, training: Vector[(Vector[Double], Vector[Double])]) = {
    val kmeansImpl = KMeansImpl(seed = 1990, k = rbfCount, eta = 1E-5, data = training.map(_._1))

    val clusters = kmeansImpl()

    val radialBasisFunctions = clusters.map { case (means, cluster) => GaussianFunction(width = cluster.map(it => kmeansImpl(it, means)).max, center = means) }

    class RadialBasisFunctionNetworkImpl()
      extends RadialBasisFunctionNetwork(
        nIn = 1,
        rbfs = radialBasisFunctions,
        nOut = 1
      ) with GradientDescent

    val net = new RadialBasisFunctionNetworkImpl()

    for (i <- 0 to 500)
      net.fit(training)
    net
  }

  val modelWith2RBF = trainAndGetModel(rbfCount = 2, trainingData)

  println(RMSE(model = modelWith2RBF, testData))
}

case class RadialBasisFunctionNetwork(nIn: Int, nOut: Int, rbfs: IndexedSeq[RadialBasisFunction]) extends RadialBasisFunctionNetworkTrait

trait RadialBasisFunctionNetworkTrait extends MachineLearningModel {
  val nIn: Int
  val nOut: Int
  val rbfs: IndexedSeq[RadialBasisFunction]
  val nHidden: Int = rbfs.length + 1
  val weights: Array[Double] = Array.fill(nHidden * nOut)(scala.math.random())

  override def getH(input: (IndexedSeq[Double], IndexedSeq[Double])): IndexedSeq[Double] = {
    rbfs.indices.map { i => rbfs(i)(input._1) }
  }

  override def apply(input: IndexedSeq[Double]): IndexedSeq[Double] = {
    for (outputIndex <- 0 until nOut) yield {
      val linearCombination = rbfs.indices.map { i => rbfs(i)(input) * weights((nHidden * outputIndex) + i) }
      val bias = weights((nHidden * outputIndex) + rbfs.length)
      linearCombination.sum + bias
    }
  }
}