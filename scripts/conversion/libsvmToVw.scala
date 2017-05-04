import java.security.MessageDigest

import org.apache.hadoop.fs.{FileSystem, Path => HadoopPath}
import org.apache.spark.rdd.RDD


def convertLine(row: String): String = {
  val elements = row.split(' ')
  val target  = elements.head.toInt * 2 - 1
  (target.toString + " |") +: elements.tail.map(e => {
    val es = e.split(':')
    val index = es(0).toInt
    if (index < 14) e
    else es.mkString("_")
  }) mkString " "
}

def md5[T](s: T) = {
  MessageDigest
    .getInstance("MD5")
    .digest(s.toString.getBytes)
    .map("%02x".format(_))
    .mkString
}

def convertFile(srcPath: String, dstPath: String) = {
  sc
    .textFile(srcPath)
    .map(convertLine)
    .zipWithIndex
    .sortBy(z => md5(z._2))
    .map(_._1)
    .saveAsTextFile(dstPath)
}

val names = List("test", "train")

def powTenAndTriple(n: Int): List[Long] = { val v = scala.math.pow(10, n).longValue; List(v, 3 * v) }
val nums = (4 to 9 flatMap powTenAndTriple).toList
def numToString(num: Long): String = num.toString.reverse.replaceAll("000", "k").reverse

val fs = FileSystem.get(sc.hadoopConfiguration)
def fileExists(path: String): Boolean = fs.exists(new HadoopPath(path))
def removeFile(path: String): Unit = fs.delete(new HadoopPath(path))

def doDataConversion = {
  for {
    num <- nums
    name <- names
    numName = numToString(num)
    srcPath = s"criteo/libsvm/$name/$numName"
    dstPath = s"criteo/vw/$name/$numName"
    if fileExists(srcPath)
    if !fileExists(dstPath + "/_SUCCESS")
  } {
    println(s"$srcPath -> $dstPath")
    removeFile(dstPath)
    convertFile(srcPath, dstPath)
  }
}


println("Use 'doDataConversion' to start data conversion.")
