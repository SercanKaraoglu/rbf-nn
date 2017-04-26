
# k-means Radial Basis Function Neural Network

This notebook consists of two section, first one is about implementation. Scala is chosen intentionally since it is perfect fit for this domain, such as, it lets you construct powerful abstractions using its properties like traits, case classes, tail recursions ..etc. Therefore, we could create a machine learning flow that each of its building block is modular. Second topic is inference which explains how we can create a model using unsupervised learning algorithm kmeans and process its output to create supervised learning model which is a radial basis function network with Gaussian Kernel. Once we done with constructing the flow we are going to try different parameters so that we can see a one model that underfit, one that overfit and fit.

## Sections

- [Implementation](#Implementation)
    - [Function Aliases](#Function Aliases)
    - [Traits](#Traits)
    - [Application](#Application)
- [Inference](#Inference)
    - [Data](#Data)
    - [RMSE on Training and Test Set](#RMSE on Training and Test Set)
    - [Chosing Models](#Chosing Models)
    - [Visualization](#Visualization)


### Implementation

Let's start with introducing the abstractions that we are going to use for the rest of the code. 

#### Function Aliases

Since the most granular abstraction is a function, here below we are going to name them based on their actual job in this implementation.


```scala211
type Features = IndexedSeq[Double]

type DistanceFunc = ((IndexedSeq[Double], IndexedSeq[Double]) => Double)
type RadialBasisFunction = (IndexedSeq[Double] => Double)
type Clustering = () => IndexedSeq[(Features, IndexedSeq[Features])]
```




    defined [32mtype[39m [36mFeatures[39m
    defined [32mtype[39m [36mDistanceFunc[39m
    defined [32mtype[39m [36mRadialBasisFunction[39m
    defined [32mtype[39m [36mClustering[39m



This definition will let us write more readable code, for example you can read above like this, 
Distance is a function that takes two point from n-dimensional space spesificed with IndexedSeq's sizes, and produces an output metric that its type is Double. Let's move forward with introducing higher level abstractions like traits.


#### Traits

You can think of a trait as a recipe for doing something, however it is not a something by itself. It is preferable over abstract classes when you want to take advantage of mixin-classes so that you can have reusable behaviours that you can use anytime to add any class that conforms to the protocol that you declare within the trait. An example to this is GradientDescent, it can actually optimize any machine learning algorithm not only neural network or something spesific. For example you can have a regression model and optimize its parameters using Gradient Descent. Here below we define those traits that are useful for this problem.


```scala211
import Math.{exp,pow,sqrt}
import scala.annotation.tailrec
import scala.util.Random

trait Optimizer {
  def fit(dataSet: Iterable[(IndexedSeq[Double], IndexedSeq[Double])], learningRate: Double)
}
trait NeuralNetworkModel extends (IndexedSeq[Double] => IndexedSeq[Double]) {
  def getH(input: (IndexedSeq[Double], IndexedSeq[Double])):IndexedSeq[Double]
}

trait EuclideanDistance extends DistanceFunc {
  override def apply(position1: IndexedSeq[Double], position2: IndexedSeq[Double]): Double = {
    val sum = position1.zip(position2).map {
      case (p1, p2) => 
        val d = p1 - p2
        pow(d,2)
    }.sum
    sqrt(sum)
  }
}

trait GradientDescent extends Optimizer with NeuralNetworkModel {
  val nHidden: Int
  val weights: Array[Double]

  def fit(trainingData: Iterable[(IndexedSeq[Double], IndexedSeq[Double])], learningRate: Double = 0.1): Unit  = {
    for(pair <- trainingData){
      val actual = this(pair._1)
      val X = getH(pair)
      val desired = pair._2
      for( outputIndex <- desired.indices) {
        val error = desired(outputIndex) - actual(outputIndex)
        val update = X.indices.map(i => learningRate * X(i) * error)
        X.indices.foreach { i =>
          val weightIndex = (nHidden  * outputIndex) + i
          weights(weightIndex) = update(i) + weights(weightIndex)
        }
        //bias term
        val weightIndex = (nHidden  * outputIndex) + X.length
        weights(weightIndex) = learningRate * error + weights(weightIndex)
      }
    }
  }
}

trait RadialBasisFunctionNetworkTrait extends NeuralNetworkModel {
  val nIn: Int
  val nOut: Int
  val rbfs: IndexedSeq[RadialBasisFunction]
  val nHidden: Int = rbfs.length + 1
  val weights: Array[Double] = Array.fill(nHidden * nOut)(scala.math.random)

  override def getH(input: (IndexedSeq[Double], IndexedSeq[Double])):IndexedSeq[Double] = {
    rbfs.indices.map { i => rbfs(i)(input._1) }
  }

  override def apply(input: IndexedSeq[Double]): IndexedSeq[Double] = {
    for (outputIndex <- 0 until nOut) yield {
      val linearCombination = rbfs.indices.map { i => rbfs(i)(input) * weights((nHidden * outputIndex) + i)}
      val bias = weights((nHidden * outputIndex) + rbfs.length)
      linearCombination.sum + bias
    }
  }
}

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

type ErrorFunc = (NeuralNetworkModel, Iterable[(IndexedSeq[Double], IndexedSeq[Double])]) => Double
```




    [32mimport [39m[36mMath.{exp,pow,sqrt}
    [39m
    [32mimport [39m[36mscala.annotation.tailrec
    [39m
    [32mimport [39m[36mscala.util.Random
    
    [39m
    defined [32mtrait[39m [36mOptimizer[39m
    defined [32mtrait[39m [36mNeuralNetworkModel[39m
    defined [32mtrait[39m [36mEuclideanDistance[39m
    defined [32mtrait[39m [36mGradientDescent[39m
    defined [32mtrait[39m [36mRadialBasisFunctionNetworkTrait[39m
    defined [32mclass[39m [36mKMeans[39m
    defined [32mtype[39m [36mErrorFunc[39m



#### Application
We can read the above code for example like this, we have NeuralNetworkModel that takes an input which consists of features as IndexedSeq and produces target values as again IndexedSeq, this is because neural networks can produce multi output like in case of multi-label problems. It defines this behaviour by extending the Function[IndexedSeq[Double],IndexedSeq[Double]] however, it also declares getH in its protocol which means that any class that implements this need to tell how to get hidden values from pair of input and target values.So any preprocessing can be done here. Consider radial basis function neural network, in this case we need to preprocess features by applying gaussian funtion and return them. Once we define this set of behaviours we can define higher-level recipes like gradient descent optimization. We use it for updating weights with back propogation. Abstract class Kmeans is also similar to trait but it needs to be first module that implementing class needs to extend. Once we create Kmeans implementation we can extend it using other traits like EuclideanDistance so that kmeans can use euclidean distance in its internal calculations. By now we are almost done, we defined our abstractions and can define actual classes. In the below we defined cases classes and object to create gaussian function.


```scala211
object GaussianFunction{
  def apply(width: Double, center: IndexedSeq[Double]): GaussianFunction = new GaussianFunction(width, center)
  def apply(widthCenterPair: (Double,IndexedSeq[Double])) = new GaussianFunction(widthCenterPair._1,widthCenterPair._2)
}

class GaussianFunction(width:Double,center: IndexedSeq[Double]) extends RadialBasisFunction {
  override def apply(x: IndexedSeq[Double]): Double = exp(-1*x.indices.map(i=>pow(x(i) - center(i), 2) / (2.0 * width * width)).sum)
}

object RMSE extends ErrorFunc {
  //RMSE = sqrt(mean((y-y_pred).^2));
  override def apply(model: NeuralNetworkModel, data: Iterable[(IndexedSeq[Double], IndexedSeq[Double])]): Double = {
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
case class KMeansImpl(override val seed: Int,override val k: Int,override val eta: Double,override val data: Vector[Features]) extends KMeans(seed, k, eta , data)
        with EuclideanDistance

case class RadialBasisFunctionNetwork(nIn: Int, nOut: Int, rbfs: IndexedSeq[RadialBasisFunction]) 
     extends RadialBasisFunctionNetworkTrait
     with GradientDescent

```




    defined [32mobject[39m [36mGaussianFunction[39m
    defined [32mclass[39m [36mGaussianFunction[39m
    defined [32mobject[39m [36mRMSE[39m
    defined [32mclass[39m [36mKMeansImpl[39m
    defined [32mclass[39m [36mRadialBasisFunctionNetwork[39m



We are all set one last thing is to load the data. Below we load file by using Scala's utility class scala.io.Source and process file read by line with regex.


```scala211
import java.io.File
val numberPattern = "([-+]?[0-9]*\\.?[0-9]+[eE][-+]?[0-9]+)?".r
val dataPath = new File("/home/foreks/git/rbf-nn/data")
def getData(path:String) = 
      scala.io.Source.fromFile(new File(dataPath, path)).getLines()
        .map { line =>
          val columns = numberPattern.findAllIn(line).filter(_ != "").map(_.toDouble).toVector
            columns.splitAt(columns.length - 1 )
          }
          .toVector
val trainingData = getData("d_reg_tra.txt")
val testData = getData("d_reg_val.txt")
```




    [32mimport [39m[36mjava.io.File
    [39m
    [36mnumberPattern[39m: [32mutil[39m.[32mmatching[39m.[32mRegex[39m = ([-+]?[0-9]*\.?[0-9]+[eE][-+]?[0-9]+)?
    [36mdataPath[39m: [32mjava[39m.[32mio[39m.[32mFile[39m = /home/foreks/git/rbf-nn/data
    defined [32mfunction[39m [36mgetData[39m
    [36mtrainingData[39m: [32mVector[39m[([32mVector[39m[[32mDouble[39m], [32mVector[39m[[32mDouble[39m])] = [33mVector[39m(
      ([33mVector[39m([32m0.88780483[39m), [33mVector[39m([32m7.0887174[39m)),
      ([33mVector[39m([32m-1.1968985[39m), [33mVector[39m([32m-6.6488336[39m)),
      ([33mVector[39m([32m-0.6805217[39m), [33mVector[39m([32m1.3097657[39m)),
      ([33mVector[39m([32m1.0287499[39m), [33mVector[39m([32m-5.4677192[39m)),
      ([33mVector[39m([32m-2.3360415[39m), [33mVector[39m([32m8.879597[39m)),
      ([33mVector[39m([32m2.0833673[39m), [33mVector[39m([32m0.5759764[39m)),
      ([33mVector[39m([32m2.7345112[39m), [33mVector[39m([32m6.9868772[39m)),
      ([33mVector[39m([32m-1.3309695[39m), [33mVector[39m([32m-14.2752[39m)),
      ([33mVector[39m([32m-1.3367205[39m), [33mVector[39m([32m-25.496883[39m)),
      ([33mVector[39m([32m1.2805311[39m), [33mVector[39m([32m2.4558209[39m)),
      ([33mVector[39m([32m1.4816721[39m), [33mVector[39m([32m-0.01585057[39m)),
    [33m...[39m
    [36mtestData[39m: [32mVector[39m[([32mVector[39m[[32mDouble[39m], [32mVector[39m[[32mDouble[39m])] = [33mVector[39m(
      ([33mVector[39m([32m1.98015[39m), [33mVector[39m([32m-6.7806596[39m)),
      ([33mVector[39m([32m-2.8219563[39m), [33mVector[39m([32m19.451004[39m)),
      ([33mVector[39m([32m-2.0240151[39m), [33mVector[39m([32m-8.6291743[39m)),
      ([33mVector[39m([32m2.1564677[39m), [33mVector[39m([32m-1.133289[39m)),
      ([33mVector[39m([32m2.918494[39m), [33mVector[39m([32m16.647087[39m)),
      ([33mVector[39m([32m0.38654075[39m), [33mVector[39m([32m-3.3443781[39m)),
      ([33mVector[39m([32m-2.9318765[39m), [33mVector[39m([32m25.607553[39m)),
      ([33mVector[39m([32m-1.4785414[39m), [33mVector[39m([32m-19.22561[39m)),
      ([33mVector[39m([32m1.3084861[39m), [33mVector[39m([32m3.079529[39m)),
      ([33mVector[39m([32m-1.9936273[39m), [33mVector[39m([32m-9.2362868[39m)),
      ([33mVector[39m([32m-2.275388[39m), [33mVector[39m([32m-2.1666518[39m)),
    [33m...[39m



Let's important what necessary for plotting graphs, and initialize.


```scala211
import $ivy.`org.plotly-scala::plotly-jupyter-scala:0.3.1`
import plotly._
import plotly.element._
import plotly.layout._
import plotly.JupyterScala._
plotly.JupyterScala.init()
```



      <script type="text/javascript">
        require.config({
  paths: {
    d3: 'https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.17/d3.min',
    plotly: 'https://cdn.plot.ly/plotly-1.12.0.min'
  },

  shim: {
    plotly: {
      deps: ['d3', 'jquery'],
      exports: 'plotly'
    }
  }
});
        

        require(['plotly'], function(Plotly) {
          window.Plotly = Plotly;
        });
      </script>
    





    [32mimport [39m[36m$ivy.$                                             
    [39m
    [32mimport [39m[36mplotly._
    [39m
    [32mimport [39m[36mplotly.element._
    [39m
    [32mimport [39m[36mplotly.layout._
    [39m
    [32mimport [39m[36mplotly.JupyterScala._
    [39m



### Inference

#### Data
In this example, we have 2 by 50 training data and 2 by 100 validation data. The end goal is to construct a regression model with two step, first unsupervised than supervised learning. First column is an input and second column is a target. We are going to use Gaussian Function as a Radial Basis Function for the neural network and use gradient descent as an optimizer. To initialize Gaussian Function's center and width we are going to run kmeans algorithm and take means as gaussian function's center and take the maximium distance between mean and cluster element as width. We are going to run the algorithm with different amount of Gaussian Units ranging from 1 to 10 and evaluate their performance on both training and test sets.


```scala211

def trainAndGetModel(rbfCount: Int, training:Vector[(Vector[Double], Vector[Double])]) = {
    val kmeansImpl = KMeansImpl(seed = 1990, k = rbfCount, eta = 1E-5, data = training.map(_._1))

    val clusters = kmeansImpl()

    val radialBasisFunctions = clusters.map { case (means, cluster) => GaussianFunction(width = cluster.map(it => kmeansImpl(it, means)).max, center = means) }

    val net = new RadialBasisFunctionNetwork(nIn = 1,rbfs = radialBasisFunctions,nOut = 1)

    for (i <- 0 to 500)
      net.fit(training,0.1)
    net
}

val models = 
    (for{i <- 1 to 10} yield { 
        ( i -> trainAndGetModel(rbfCount=i,trainingData))
    })
```




    defined [32mfunction[39m [36mtrainAndGetModel[39m
    [36mmodels[39m: [32mcollection[39m.[32mimmutable[39m.[32mIndexedSeq[39m[([32mInt[39m, [32mRadialBasisFunctionNetwork[39m)] = [33mVector[39m(
      ([32m1[39m, <function1>),
      ([32m2[39m, <function1>),
      ([32m3[39m, <function1>),
      ([32m4[39m, <function1>),
      ([32m5[39m, <function1>),
      ([32m6[39m, <function1>),
      ([32m7[39m, <function1>),
      ([32m8[39m, <function1>),
      ([32m9[39m, <function1>),
      ([32m10[39m, <function1>)
    )



#### Visualization

In the above we defined radial basis function with 1 input 1 output and k units of gaussian expert and trained it with 500 iterations with 0.1 learningRate. We could have had different results by adjusting learning rate, however 0.1 produced the minimum root mean square error so we choose it.


```scala211
def plotBarGraphForErrors(models:IndexedSeq[(Int, NeuralNetworkModel)], 
                          data: Vector[(Vector[Double], Vector[Double])],
                          label: String) = {
    val errors = models.map(model=>(model._1.toString, RMSE(model._2, data))).toSeq

    val (x, y) = errors.unzip

    Bar(x, y).plot(title=label)
}
```




    defined [32mfunction[39m [36mplotBarGraphForErrors[39m




```scala211
plotBarGraphForErrors(models, trainingData, "Training Data RMSE")
```


<div class="chart" id="plot-1478169693"></div>







    [36mres7[39m: [32mString[39m = [32m"plot-1478169693"[39m




```scala211
plotBarGraphForErrors(models, testData, "Test Data RMSE")
```


<div class="chart" id="plot-1863303345"></div>







    [36mres8[39m: [32mString[39m = [32m"plot-1863303345"[39m



#### Chosing Models

By looking at the above results, we can pick three different models one that underfits, one that overfits and one that good fits. When RBF unit count is 1 it is seen that error is high for both training and test data, so we can assume that this is a underfit model. When RBF unit count is 3, error drops significantly on the test data and it also drops on the training data consistently as RBF gets higher. So we can pick a model with RBF count 3 as a good fit model. However as the RBF count increases, error gets higher in the test data while keep dropping in the training data. This gives us hint that as the model get more complex with more RBF units it tends to become overfit. So we can pick a model with RBF count 10 as an example of overfitting model.


```scala211
val underfitModel = models(0)._2
val goodFitModel = models(2)._2
val overfitModel = models(9)._2
```




    [36munderfitModel[39m: [32mRadialBasisFunctionNetwork[39m = <function1>
    [36mgoodFitModel[39m: [32mRadialBasisFunctionNetwork[39m = <function1>
    [36moverfitModel[39m: [32mRadialBasisFunctionNetwork[39m = <function1>



Let's plot predictions vs actual data for "training data".


```scala211
def plotLineGraphForInputAndOutput(model: NeuralNetworkModel,  
                                            data: Vector[(Vector[Double], Vector[Double])],
                                            label: String) = {
    val prediction = data.map(_._1).map(model)

    val indices = prediction.indices

    val plot = Seq(
          Scatter(indices, prediction.map(_(0)),name="model output"),
          Scatter(indices, data.flatMap(_._1),name="model input")
    )

    plot.plot(
        title = label
    )
}
```




    defined [32mfunction[39m [36mplotLineGraphForInputAndOutput[39m




```scala211
plotLineGraphForInputAndOutput(underfitModel, trainingData, "Underfitting Model Input and Output for Training Data")
```


<div class="chart" id="plot-1428836905"></div>







    [36mres11[39m: [32mString[39m = [32m"plot-1428836905"[39m




```scala211
plotLineGraphForInputAndOutput(goodFitModel,trainingData,"Good-Fitting Model Input and Output for Training Data")
```


<div class="chart" id="plot-455821534"></div>







    [36mres12[39m: [32mString[39m = [32m"plot-455821534"[39m




```scala211
plotLineGraphForInputAndOutput(overfitModel,trainingData,"Overfitting Model Input and Output for Training Data")
```


<div class="chart" id="plot-153396448"></div>







    [36mres13[39m: [32mString[39m = [32m"plot-153396448"[39m



When we look at the graphs, we conclude that underfitting model produces output close to random around zero so doesn't model well even on training data. Overfitting model is more responsive to the noise in the training data, however best fit model seem to generalize well in terms of response to noise and actual movements.

Now we are going to take a look at the RBF outputs, weighted values and actual data together on the same graph for each case underfitting, overfitting and good-fit.


```scala211
def plotLineGraphForRBFPredictionAndActualData(model: NeuralNetworkModel,  
                                               data: Vector[(Vector[Double], Vector[Double])],
                                              label: String) = {
    val prediction = data.map(_._1).map(model)
    val rbfOutputs =  data.map(model.getH)
    val rbfCount =  rbfOutputs.head.size
    
    val indices = data.indices

    val plot = Seq(
          Scatter(indices, prediction.map(_(0)),name="prediction"),
          Scatter(indices, data.flatMap(_._2),name="actual")
    ) ++ (0 until rbfCount).map(rbfIndex => Scatter(indices, rbfOutputs.map(_(rbfIndex)),name=s"RBOutput$rbfIndex")).toSeq

    plot.plot(
        title = label
    )
}
```




    defined [32mfunction[39m [36mplotLineGraphForRBFPredictionAndActualData[39m




```scala211
val allData =  trainingData ++ testData
plotLineGraphForRBFPredictionAndActualData(underfitModel, allData, "Underfitting Model Outputs vs Actual")
```


<div class="chart" id="plot-125939994"></div>







    [36mallData[39m: [32mVector[39m[([32mVector[39m[[32mDouble[39m], [32mVector[39m[[32mDouble[39m])] = [33mVector[39m(
      ([33mVector[39m([32m0.88780483[39m), [33mVector[39m([32m7.0887174[39m)),
      ([33mVector[39m([32m-1.1968985[39m), [33mVector[39m([32m-6.6488336[39m)),
      ([33mVector[39m([32m-0.6805217[39m), [33mVector[39m([32m1.3097657[39m)),
      ([33mVector[39m([32m1.0287499[39m), [33mVector[39m([32m-5.4677192[39m)),
      ([33mVector[39m([32m-2.3360415[39m), [33mVector[39m([32m8.879597[39m)),
      ([33mVector[39m([32m2.0833673[39m), [33mVector[39m([32m0.5759764[39m)),
      ([33mVector[39m([32m2.7345112[39m), [33mVector[39m([32m6.9868772[39m)),
      ([33mVector[39m([32m-1.3309695[39m), [33mVector[39m([32m-14.2752[39m)),
      ([33mVector[39m([32m-1.3367205[39m), [33mVector[39m([32m-25.496883[39m)),
      ([33mVector[39m([32m1.2805311[39m), [33mVector[39m([32m2.4558209[39m)),
      ([33mVector[39m([32m1.4816721[39m), [33mVector[39m([32m-0.01585057[39m)),
    [33m...[39m
    [36mres15_1[39m: [32mString[39m = [32m"plot-125939994"[39m




```scala211
plotLineGraphForRBFPredictionAndActualData(overfitModel, allData, "OverFitting Model Outputs vs Actual")
```


<div class="chart" id="plot-891399789"></div>







    [36mres16[39m: [32mString[39m = [32m"plot-891399789"[39m




```scala211
plotLineGraphForRBFPredictionAndActualData(goodFitModel, allData, "GoodFitting Model Outputs vs Actual")
```


<div class="chart" id="plot-1299780919"></div>







    [36mres17[39m: [32mString[39m = [32m"plot-1299780919"[39m


