package org.bau.ann

import scala.annotation.tailrec
import scala.util.Random

/**
  * Created by foreks on 22.04.2017.
  */
abstract class KMeans(val seed: Int, val k: Int, val eta: Double, val data: Vector[Features]) extends DistanceFunc with Clustering {
  val rand = new Random(seed)
  val _data = data
  val means: Vector[Features] = (0 until k).map(_ => data(rand.nextInt(data.length))).toVector

  def apply(): IndexedSeq[(Features, IndexedSeq[Features])] = {
    @tailrec
    def _kmeans(means: IndexedSeq[Features]): IndexedSeq[(Features, IndexedSeq[Features])] = {
      val newMeans: IndexedSeq[(Features, IndexedSeq[Features])] =
        means.map(mean => {
          _data.groupBy(p => means.minBy(m => this (m, p))).get(mean) match {
            case Some(c) => (c.transpose.map(_.sum).map(x => x / c.length), c)
            case _ => (mean, Vector(mean))
          }
        })
      val converged = (means zip newMeans.map(_._1)).forall {
        case (oldMean, newMean) => this (oldMean, newMean) <= eta
      }
      if (!converged) _kmeans(newMeans.map(_._1)) else newMeans
    }

    _kmeans(means)
  }
}
