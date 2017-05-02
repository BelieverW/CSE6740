import org.apache.commons.math3.linear._
import scala.util.control.Breaks._

import scala.io.Source


object LocalALS {

  // Parameters set through command line arguments
  var M = 9066 // Number of movies
  var U = 671 // Number of users
  var ITERATIONS = 20

  def main(args: Array[String]) {


    val TrainR = generateR("./small_dataset/TrainData.csv")
    val TestR = generateR("./small_dataset/TestData.csv")

    val F = 15
    val LAMBDA = 0.00025
    val la = List(0)



    for (LAMBDA <- la) {
      println(s"Running with M=$M, U=$U, F=$F, iters=$ITERATIONS")
      // Initialize m and u randomly
      var ms = Array.fill(M)(randomVector(F))
      var us = Array.fill(U)(randomVector(F))

      // Iteratively update movies then users
      for (iter <- 1 to ITERATIONS) {
        println(s"Iteration $iter:")
        ms = (0 until M).map(i => updateMovie(i, ms(i), us, TrainR, F, LAMBDA)).toArray
        us = (0 until U).map(j => updateUser(j, us(j), ms, TrainR, F, LAMBDA)).toArray
        val rmse_val = rmse(TrainR, ms, us) - 2
        println("RMSE = " + rmse_val)
        scala.tools.nsc.io.File(s"res_output/iteration_train").appendAll(s"$rmse_val \n")
        val test_rmse = rmse(TestR, ms, us) - 2
        val filename = s"res_output/iteration_test"
        scala.tools.nsc.io.File(filename).appendAll(s"$test_rmse \n")
      }
    }
  }

  def generateR(path: String): RealMatrix = {
    val A = new Array2DRowRealMatrix(9066, 671)
    val user_rating = readCSV(path)
    for(i <- 0 until A.getRowDimension())
      for(j <- 0 until A.getColumnDimension())
        A.setEntry(i, j, user_rating(j)(i))
    A
  }

  def readCSV(path: String) : Array[Array[Double]] = {
    Source.fromFile(path)
      .getLines()
      .map(_.split(",").map(_.trim.toDouble))
      .toArray
  }

  def rmse(targetR: RealMatrix, ms: Array[RealVector], us: Array[RealVector]): Double = {
    val r = new Array2DRowRealMatrix(M, U)
    for (i <- 0 until M; j <- 0 until U) {
      r.setEntry(i, j, ms(i).dotProduct(us(j)))
    }
    val diffs = r.subtract(targetR)
    var sumSqs = 0.0
    var cnt = 0
    for (i <- 0 until M; j <- 0 until U) {
      if (targetR.getEntry(i, j) != 0) {
        val diff = diffs.getEntry(i, j)
        sumSqs += diff * diff
        cnt += 1;
      }
    }
    // math.sqrt(sumSqs / (M.toDouble * U.toDouble))
    math.sqrt(sumSqs / cnt)
  }

  def updateMovie(i: Int, m: RealVector, us: Array[RealVector], R: RealMatrix, F:Int, LAMBDA: Double) : RealVector = {
    var XtX: RealMatrix = new Array2DRowRealMatrix(F, F)
    var Xty: RealVector = new ArrayRealVector(F)
    // For each user that rated the movie
    for (j <- 0 until U) {
      val u = us(j)
      // Add u * u^t to XtX
      XtX = XtX.add(u.outerProduct(u))
      // Add u * rating to Xty
      Xty = Xty.add(u.mapMultiply(R.getEntry(i, j)))
    }
    // Add regularization coefficients to diagonal terms
    for (d <- 0 until F) {
      XtX.addToEntry(d, d, LAMBDA * U)
    }
    // Solve it with Cholesky
    new CholeskyDecomposition(XtX).getSolver.solve(Xty)
  }

  def updateUser(j: Int, u: RealVector, ms: Array[RealVector], R: RealMatrix, F: Int, LAMBDA: Double) : RealVector = {
    var XtX: RealMatrix = new Array2DRowRealMatrix(F, F)
    var Xty: RealVector = new ArrayRealVector(F)
    // For each movie that the user rated
    for (i <- 0 until M) {
      val m = ms(i)
      // Add m * m^t to XtX
      XtX = XtX.add(m.outerProduct(m))
      // Add m * rating to Xty
      Xty = Xty.add(m.mapMultiply(R.getEntry(i, j)))
    }
    // Add regularization coefficients to diagonal terms
    for (d <- 0 until F) {
      XtX.addToEntry(d, d, LAMBDA * M)
    }
    // Solve it with Cholesky
    new CholeskyDecomposition(XtX).getSolver.solve(Xty)
  }

  private def randomVector(n: Int): RealVector =
    new ArrayRealVector(Array.fill(n)(math.random))

  private def randomMatrix(rows: Int, cols: Int): RealMatrix =
    new Array2DRowRealMatrix(Array.fill(rows, cols)(math.random))

}