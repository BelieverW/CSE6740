package edu.gatech.cse6740.recommender

import scala.io.Source
import org.apache.commons.math3.linear._

object TestMatrixBuilder {
  def main(arg : Array[String]): Unit = {
    val rt = readCSV()
    println(rt(1)(1))
  }
  def readCSV() : Array[Array[Double]] = {
    Source.fromFile("./small_dataset/users_ratings.txt")
      .getLines()
      .map(_.split(",").map(_.trim.toDouble))
      .toArray
  }
}

//671, 9066