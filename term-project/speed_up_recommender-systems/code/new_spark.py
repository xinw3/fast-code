import sys
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, Rating

data_dir = os.path.join('../', 'RSdata/')
training_file = os.path.join(data_dir, "training_rating.dat")
test_file = os.path.join(data_dir, "testing.dat")
output_file = os.path.join('./', "results.csv")
removed_training_file = os.path.join(data_dir, "output_file.dat")

def loadMovieNames():
    movieNames = {}
    with open("ml-100k/u.ITEM") as f:
        for line in f:
            fields = line.split('::')
            movieNames[int(fields[0])] = fields[1].decode('ascii', 'ignore')
    return movieNames

conf = SparkConf().setMaster("local[*]").setAppName("MovieRecommendationsALS")
sc = SparkContext(conf = conf)

print "\nLoading movie names..."
nameDict = loadMovieNames()

data = sc.textFile("removed_training_file")

ratings = data.map(lambda l: l.split()).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2]))).cache()

# Build the recommendation model using Alternating Least Squares
print "\nTraining recommendation model..."
rank = 10
numIterations = 20
model = ALS.train(ratings, rank, numIterations)

userID = int(sys.argv[1])

print "\nRatings for user ID " + str(userID) + ":"
userRatings = ratings.filter(lambda l: l[0] == userID)
for rating in userRatings.collect():
    print nameDict[int(rating[1])] + ": " + str(rating[2])

print "\nTop 10 recommendations:"
recommendations = model.recommendProducts(userID, 10)
for recommendation in recommendations:
    print nameDict[int(recommendation[1])] + \
        " score " + str(recommendation[2])