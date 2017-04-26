package org.bau.ann

/**
  * Created by foreks on 22.04.2017.
  */
trait EuclideanDistance extends DistanceFunc {
  override def apply(position1: IndexedSeq[Double], position2: IndexedSeq[Double]): Double = {
    val sum = position1.zip(position2).map {
      case (p1, p2) =>
        val d = p1 - p2
        d * d
    }.sum
    Math.sqrt(sum)
  }
}
