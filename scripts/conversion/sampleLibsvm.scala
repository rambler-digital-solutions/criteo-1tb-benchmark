type Data = (org.apache.spark.rdd.RDD[String], Long)

def sample(data: Data, n: Long): Data = {
  val rdd = data._1
  val count = data._2
  val ratio = n.toDouble / count
  val sampledRdd = rdd.sample(false, scala.math.min(1.0, ratio * 1.01), scala.util.Random.nextLong)
  val exactSampledRdd = sampledRdd.zipWithIndex.filter { case (_, i) => i < n } map (_._1)
  val exactCount = exactSampledRdd.count
  (exactSampledRdd, exactCount)
}

def load(path: String): (Data, Data) = {
  val filesAndStats = 0 to 23 map {
    case i => {
      val dayPath = s"$path/day_${i}.*"
      val rdd = sc.textFile(dayPath)
      val count = rdd.count
      (rdd, count)
    }
  }
  val test = filesAndStats.last
  val train = filesAndStats.init.reduce((a, b) => (a._1 union b._1, a._2 + b._2))
  (train, test)
}

val fs = org.apache.hadoop.fs.FileSystem.get(sc.hadoopConfiguration)

val numberOfParts = 1024

def writeSamples(data: Data, samples: List[Long], path: String, ext: String): Unit = samples foreach {
  case n =>
    val name = n.toString.reverse.replaceAll("000", "k").reverse
    val writePath = s"$path/$name.$ext"
    val hadoopSuccessPath = new org.apache.hadoop.fs.Path(writePath + "/_SUCCESS")
    if (fs.exists(hadoopSuccessPath)) {
      println(s"Data was already successfully written to $writePath, skipping.")
    } else {
      val hadoopPath = new org.apache.hadoop.fs.Path(writePath)
      println(s"Removing $writePath.")
      fs.delete(hadoopPath)
      println("Sampling data")
      val sampledData = sample(data, n)
      println(s"Writing ${sampledData._2} lines to $writePath.")
      sampledData._1.coalesce(numberOfParts).saveAsTextFile(writePath)
    }
}

def powTenAndTriple(n: Int): List[Long] = { val v = scala.math.pow(10, n).longValue; List(v, 3 * v) }

val testSamples = List(1000000l)
val trainSamples = (4 to 9 flatMap powTenAndTriple).toList

def processDataPersist(what: String): Unit = {
  println(s"Working with $what.")

  val dataPath = s"criteo_1tb/$what"
  println(s"Loading data from $dataPath.")
  val (train, test) = load(dataPath)
  println("Data loaded.")

  def processDataSet(name: String, data: Data, samples: List[Long]): Unit = {
    println(s"Sampling $name to ${samples.mkString("[", ", ", "]")} lines.")
    writeSamples(data, samples, s"$dataPath/$name", what)
  }

  test._1.persist
  processDataSet("test", test, testSamples)
  test._1.unpersist(true)

  train._1.persist
  processDataSet("train", train, trainSamples)
  train._1.unpersist(true)

  println(s"Done with $what.")
}

def doDataPreparationLibSVM = {
  processDataPersist("libsvm")
}

println("Use 'doDataPreparationLibSVM' to start data preparation.")
