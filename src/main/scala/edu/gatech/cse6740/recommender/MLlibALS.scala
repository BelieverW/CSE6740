package edu.gatech.cse6740.recommender


import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.{ALS, Rating}

object MLlibALS {
  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setMaster("local[*]").setAppName("cf user-based")
    val sc = new SparkContext(sparkConf)

    // extract (userid, itemid, rating) from ratings data
    val ratings = sc.textFile("data/ratings.dat").map(line => {
      val fields = line.split("::")
      Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
    }).cache()

    val users = ratings.map(_.user).distinct()
    val products = ratings.map(_.product).distinct()
    println("Got "+ratings.count()+" ratings from "+users.count+" users on "+products.count+" products.")


    val ranks = List(500)
    val lambdas = List(0.015)
    val numIters = List(20)

    var res = List("res")

    for (lambda <- lambdas;rank <- ranks) {
      val numIter = 15
      val model = ALS.train(ratings, rank, numIter, lambda)

      val usersProducts= ratings.map { case Rating(user, product, rate) =>
        (user, product)
      }

      var predictions = model.predict(usersProducts).map { case Rating(user, product, rate) =>
        ((user, product), rate)
      }

      val ratesAndPreds = ratings.map { case Rating(user, product, rate) =>
        ((user, product), rate)
      }.join(predictions)

      val rmse= math.sqrt(ratesAndPreds.map { case ((user, product), (r1, r2)) =>
        val err = (r1 - r2)
        err * err
      }.mean())

      val curRes = s"rank = $rank, lambda = $lambda, numIter = $numIter, RMSE = $rmse \n"
      println(curRes)
      scala.tools.nsc.io.File("res.txt").appendAll(curRes)
      res = res:::List(curRes)

      //println(s"RMSE = $rmse")
    }
    res.foreach(println)

//    val rank = 3
//    val lambda = 0.01
//    val numIterations = 20
//    val model = ALS.train(ratings, rank, numIterations, lambda)

//    val usersProducts= ratings.map { case Rating(user, product, rate) =>
//      (user, product)
//    }
//
//    var predictions = model.predict(usersProducts).map { case Rating(user, product, rate) =>
//      ((user, product), rate)
//    }
//
//    val ratesAndPreds = ratings.map { case Rating(user, product, rate) =>
//      ((user, product), rate)
//    }.join(predictions)
//
//    val rmse= math.sqrt(ratesAndPreds.map { case ((user, product), (r1, r2)) =>
//      val err = (r1 - r2)
//      err * err
//    }.mean())
//
//    println(s"RMSE = $rmse")

  }
}