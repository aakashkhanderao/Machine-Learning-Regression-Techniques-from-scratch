using CSV
using Statistics
using DataFrames

dataset=CSV.read("data/housingPriceData.csv")
bedrooms=dataset.bedrooms

m = length(bedrooms)
trainDataCount=convert(Int64,round(m*0.8))
testDataCount=convert(Int64,m-trainDataCount)

#training data set
bedroomsTrain=bedrooms[1:trainDataCount]
sqft_livingTrain=dataset.sqft_living[1:trainDataCount]

priceTrain=dataset.price[1:trainDataCount]

#test data set
bedroomsTest=bedrooms[trainDataCount+1:m]
sqft_livingTest=dataset.sqft_living[trainDataCount+1:m]

priceTest=dataset.price[trainDataCount+1:m]


#standardised training dataset
bedroomsTrainsq=((bedroomsTrain.^2).-mean(bedroomsTrain.^2))./std(bedroomsTrain.^2)
sqft_livingTrainsq=((sqft_livingTrain.^2).-mean(sqft_livingTrain.^2))./std(sqft_livingTrain.^2)
productTrain=((bedroomsTrain.*sqft_livingTrain.^2).-mean(bedroomsTrain.*sqft_livingTrain.^2))./std(bedroomsTrain.*sqft_livingTrain.^2)

bedroomsTrain=(bedroomsTrain.-mean(bedroomsTrain))./std(bedroomsTrain)
sqft_livingTrain=(sqft_livingTrain.-mean(sqft_livingTrain))./std(sqft_livingTrain)


#standardised test dataset
bedroomsTestsq=((bedroomsTest.^2).-mean(bedroomsTest.^2))./std(bedroomsTest.^2)
sqft_livingTestsq=((sqft_livingTest.^2).-mean(sqft_livingTest.^2))./std(sqft_livingTest.^2)
productTest=((bedroomsTest.*sqft_livingTest.^2).-mean(bedroomsTest.*sqft_livingTest.^2))./std(bedroomsTest.*sqft_livingTest.^2)

bedroomsTest=(bedroomsTest.-mean(bedroomsTest))./std(bedroomsTest)
sqft_livingTest=(sqft_livingTest.-mean(sqft_livingTest))./std(sqft_livingTest)

# X, Y of train dataset
len=length(bedroomsTrain)
x0 = ones(len)
X_train = cat(x0, bedroomsTrain, sqft_livingTrain,bedroomsTrainsq, sqft_livingTrainsq, productTrain,dims=2)
Y_train=cat(priceTrain,dims=2)


# X, Y of test dataset
len=length(bedroomsTest)
x0 = ones(len)
X_test=cat(x0, bedroomsTest,sqft_livingTest,bedroomsTestsq, sqft_livingTestsq,productTest, dims=2)
Y_test=cat(priceTest,dims=2)

# Define a function to calculate cost function
function costFunction(X, Y, B)
    m = length(Y)
    cost = sum(((X * B) - Y).^2)/(2*m)
    return cost
end

# # Initial coefficients
B = ones(6, 1)
# Calcuate the cost with intial model parameters B=[0,0,0,0]
intialCost = costFunction(X_train, Y_train, B)

# Define a function to perform gradient descent
function gradientDescent(X, Y, B, learningRate, numIterations)
    costHistory = zeros(numIterations)
    m = length(Y)
    # do gradient descent for require number of iterations
    for iteration in 1:numIterations
        # Predict with current model B and find loss
        loss = (X * B) - Y
        # Compute Gradients: Ref to Andrew Ng. course notes linked on course page and Moodle
        gradient = (X' * loss)/m
        # Perform a descent step in direction oposite to gradient; we want to minimize cost!
        B = B - learningRate * gradient
        # Calculate cost of the new model found by descending a step above
        cost = costFunction(X, Y, B)
        # Store costs in a vairable to visualize later
        costHistory[iteration] = cost
    end
    return B, costHistory
end

#specify learning rate and number of iterations
learningRate = 0.01
newB, costHistory = gradientDescent(X_train, Y_train, B, learningRate, 1000)

# Make predictions using the learned model; newB
YPred_train = X_train * newB
YPred_test = X_test * newB
# print(newB)

#RMS CALCULATION for training data
# print("RMS error of training data----")
RMSE=sqrt(sum(((YPred_train.-Y_train).^2))/len)
# print(RMSE)

#RMS CALCULATION for test data
print("RMS error of test data-------")
RMSE=sqrt(sum(((YPred_test.-Y_test).^2))/len)
print(RMSE)

#R square value for training data
# print("R square of training data----")
R2=1-(sum(((YPred_train.-Y_train).^2))/sum((Y_train.-mean(Y_train)).^2))
# print(R2)

#R square value for test data
print("R square error of test data----")
R2=1-(sum(((YPred_test.-Y_test).^2))/sum((Y_test.-mean(Y_test)).^2))
print(R2)

YPred=[YPred_train ;YPred_test]
CSV.write("data/1b.csv",DataFrame(YPred), writeheader=false)
