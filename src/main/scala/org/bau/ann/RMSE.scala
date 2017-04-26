package org.bau.ann

import Math.{pow,sqrt}

/**
  * Created by foreks on 25.04.2017.
  */
object RMSE extends ErrorFunc {
  //RMSE = sqrt(mean((y-y_pred).^2));
  override def apply(model: Function[IndexedSeq[Double], IndexedSeq[Double]], data: Iterable[(IndexedSeq[Double], IndexedSeq[Double])]): Double = {
    var length = 1d
    val sumOfSquareError = (for {pair <- data.map(_._1).map(model).zip(data.map(_._2))
                                 actual = pair._1
                                 desired = pair._2
    } yield {
      length+=1
      pow(actual(0) - desired(0), 2)
    }).sum
    val meanSquareError = sumOfSquareError / length

    sqrt(meanSquareError)
  }
}
