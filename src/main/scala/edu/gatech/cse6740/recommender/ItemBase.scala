package edu.gatech.cse6740.recommender

import edu.gatech.cse6740.similarity.Similarity
import org.apache.spark.{SparkConf, SparkContext}



/**
  * @author Huanli Wang <hlwang@gatech.edu>
  * */

object ItemBase {

  def main(args: Array[String]) {


    val sparkConf = new SparkConf().setMaster("local[*]").setAppName("cf item-based")
    val sc = new SparkContext(sparkConf)


    // extract (userid, itemid, rating) from ratings data
    val oriRatings = sc.textFile("small_dataset/ratings.csv").map(line => {
      val fields = line.split(",")
      (fields(0).toLong, fields(1).toLong, fields(2).toDouble)
    })

    //filter redundant (user,item,rating),this set user favorite (best-loved) 100 item
    val ratings = oriRatings.groupBy(k=>k._1).flatMap(x=>(x._2.toList.sortWith((x,y)=>x._3>y._3).take(100)))
    // ratings.take(5).foreach(println)


    // get num raters per movie, keyed on item id,,item2manyUser formating as (item,(user,item,rating))
    val item2manyUser = ratings.groupBy(tup => tup._2)
    val numRatersPerItem = item2manyUser.map(grouped => (grouped._1, grouped._2.size))
    numRatersPerItem.take(5).foreach(println)

    // join ratings with num raters on item id,,ratingsWithSize formating as (user,item,rating,numRaters)
    val ratingsWithSize = item2manyUser.join(numRatersPerItem)
      .flatMap(joined => {
        joined._2._1.map(f => (f._1, f._2, f._3, joined._2._2))
      })

    // ratingsWithSize now contains the following fields: (user, item, rating, numRaters).

    // dummy copy of ratings for self join ,formating as (user, (user, item, rating, numRaters))
    val ratings2 = ratingsWithSize.keyBy(tup => tup._1)


    // join on userid and filter item pairs such that we don't double-count and exclude self-pairs

    // ratings2.join(ratings2) will result in (user, ((user, item, rating, numRaters), (user, item, rating, numRaters)))
    val ratingPairs =ratings2.join(ratings2).filter(f => f._2._1._2 < f._2._2._2)
    // ratingPairs will be (user, ((.., item1, ..), (.., item2, ..))) where item1 < item2 it's a kind of combination

    // compute raw inputs to similarity metrics for each item pair

    val tempVectorCalcs =
      ratingPairs.map(data => {
        val key = (data._2._1._2, data._2._2._2)
        val stats =
          (data._2._1._3 * data._2._2._3, // rating 1 * rating 2
            data._2._1._3,                // rating item 1
            data._2._2._3,                // rating item 2
            math.pow(data._2._1._3, 2),   // square of rating item 1
            math.pow(data._2._2._3, 2),   // square of rating item 2
            data._2._1._4,                // number of raters item 1
            data._2._2._4)                // number of raters item 2
        (key, stats)
      })

    val vectorCalcs = tempVectorCalcs.groupByKey().map(data => {
      val key = data._1
      val vals = data._2
      val size = vals.size
      val dotProduct = vals.map(f => f._1).sum
      val ratingSum = vals.map(f => f._2).sum
      val rating2Sum = vals.map(f => f._3).sum
      val ratingSq = vals.map(f => f._4).sum
      val rating2Sq = vals.map(f => f._5).sum
      val numRaters = vals.map(f => f._6).max
      val numRaters2 = vals.map(f => f._7).max
      (key, (size, dotProduct, ratingSum, rating2Sum, ratingSq, rating2Sq, numRaters, numRaters2))
    })
    //.filter(x=>x._2._1>1)

    val inverseVectorCalcs = vectorCalcs.map(x=>((x._1._2,x._1._1),(x._2._1,x._2._2,x._2._4,x._2._3,x._2._6,x._2._5,x._2._8,x._2._7)))
    val vectorCalcsTotal = vectorCalcs ++ inverseVectorCalcs


    // compute similarity metrics for each item pair
    // modify formula as : cosSim *size/(numRaters*math.log10(numRaters2+10))
    val tempSimilarities =
    vectorCalcsTotal.map(fields => {
      val key = fields._1
      val (size, dotProduct, ratingSum, rating2Sum, ratingNormSq, rating2NormSq, numRaters, numRaters2) = fields._2
//      val cosSim = Similarity.cosineSimilarity(dotProduct, scala.math.sqrt(ratingNormSq), scala.math.sqrt(rating2NormSq))*size/(numRaters*math.log10(numRaters2+10))
//      (key._1,(key._2, cosSim))
              val corr = Similarity.correlation(size, dotProduct, ratingSum, rating2Sum, ratingNormSq, rating2NormSq)
              (key._1,(key._2, corr))
    })

    val similarities = tempSimilarities.groupByKey().flatMap(x=>{
      x._2.map(temp=>(x._1,(temp._1,temp._2))).toList.sortWith((a,b)=>a._2._2>b._2._2).take(50)
    })


    // ratins is (user,item,rating)
    // ratingsInverse format (item,(user,raing))
    val ratingsInverse = ratings.map(rating=>(rating._2,(rating._1,rating._3)))

    // ratingsInverse.join(similarities) will be formatting as (item1 ((user, rating), (item2, cosSim)))
    //  statistics format ((user,item),(sim,sim*rating))
    val statistics = ratingsInverse.join(similarities).map(x=>((x._2._1._1,x._2._2._1),(x._2._2._2,x._2._1._2*x._2._2._2)))

    // after reduceByKey it'll be ((user, item), (sum(sim), sum(sim*rating)))
    // predictResult fromat ((user,item),predict)
    val predictResult = statistics.reduceByKey((x,y)=>((x._1+y._1),(x._2+y._2))).map(x=>(x._1,x._2._2/x._2._1))

    //      val predictResult = statistics.reduceByKey((x,y)=>((x._1+y._1),(x._2+y._2))).map(x=>(x._1,x._2._2))

    // oriRatings will be (userid, itemid, rating)
    val filterItem = oriRatings.map(x=>((x._1,x._2),Double.NaN))
    val totalScore = predictResult ++ filterItem

    //      val temp = totalScore.reduceByKey(_+_)

    val finalResult = totalScore.reduceByKey(_+_).filter(x=> !(x._2 equals(Double.NaN))).
      map(x=>(x._1._1,x._1._2,x._2)).groupBy(x=>x._1).flatMap(x=>(x._2.toList.sortWith((a,b)=>a._3>b._3).take(50)))


    val ratesAndPreds = oriRatings.map{record => ((record._1, record._2), record._3)}.join(predictResult)

    val rmse= math.sqrt(ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      val err = (r1 - r2)
      err * err
    }.mean())

    println(s"RMSE = $rmse")
  }

}