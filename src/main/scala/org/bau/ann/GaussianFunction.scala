package org.bau.ann
import Math.{exp,pow}
/**
  * Created by sercan on 18.04.2017.
  */
object GaussianFunction{
  def apply(width: Double, center: IndexedSeq[Double]): GaussianFunction = new GaussianFunction(width, center)
  def apply(widthCenterPair: (Double,IndexedSeq[Double])) = new GaussianFunction(widthCenterPair._1,widthCenterPair._2)
}
class GaussianFunction(width:Double,center: IndexedSeq[Double]) extends RadialBasisFunction {
  override def apply(x: IndexedSeq[Double]): Double = exp(-1*x.indices.map(i=>pow(x(i) - center(i), 2) / (2.0 * width * width)).sum)
}