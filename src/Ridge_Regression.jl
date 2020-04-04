using CSV
using Statistics
using DataFrames

dataset=CSV.read("data/housingPriceData.csv")
bedrooms=dataset.bedrooms

m = length(bedrooms)
trainDataCount=convert(Int64,round(m*0.6))
validationDataCount=convert(Int64,round(m*0.2))
testDataCount=convert(Int64,round(m*0.2))

#training dataset
bedroomsTrain=bedrooms[1:trainDataCount]
bathroomsTrain=dataset.bathrooms[1:trainDataCount]
sqft_livingTrain=dataset.sqft_living[1:trainDataCount]
priceTrain=dataset.price[1:trainDataCount]

#validation dataset
bedroomsVal=bedrooms[trainDataCount+1:trainDataCount+validationDataCount]
bathroomsVal=dataset.bathrooms[trainDataCount+1:trainDataCount+validationDataCount]
sqft_livingVal=dataset.sqft_living[trainDataCount+1:trainDataCount+validationDataCount]
priceVal=dataset.price[trainDataCount+1: trainDataCount+validationDataCount]

#test dataset
bedroomsTest=bedrooms[trainDataCount+validationDataCount+1:m]
bathroomsTest=dataset.bathrooms[trainDataCount+validationDataCount+1:m]
sqft_livingTest=dataset.sqft_living[trainDataCount+validationDataCount+1:m]
priceTest=dataset.price[trainDataCount+validationDataCount+1:m]


#standardised training dataset
bedroomsTrain=(bedroomsTrain.-mean(bedroomsTrain))./std(bedroomsTrain)
bathroomsTrain=(bathroomsTrain.-mean(bathroomsTrain))./std(bathroomsTrain)
sqft_livingTrain=(sqft_livingTrain.-mean(sqft_livingTrain))./std(sqft_livingTrain)

#standardised validation dataset
bedroomsVal=(bedroomsVal.-mean(bedroomsVal))./std(bedroomsVal)
bathroomsVal=(bathroomsVal.-mean(bathroomsVal))./std(bathroomsVal)
sqft_livingVal=(sqft_livingVal.-mean(sqft_livingVal))./std(sqft_livingVal)

#standardised test dataset
bedroomsTest=(bedroomsTest.-mean(bedroomsTest))./std(bedroomsTest)
bathroomsTest=(bathroomsTest.-mean(bathroomsTest))./std(bathroomsTest)
sqft_livingTest=(sqft_livingTest.-mean(sqft_livingTest))./std(sqft_livingTest)



#X, Y of training dataset 
len = length(bedroomsTrain)
x0 = ones(len)

X_train = cat(x0, bedroomsTrain, bathroomsTrain,sqft_livingTrain, dims=2)
Y_train = cat(priceTrain,dims=2)


#X, Y of validation dataset
len = length(bedroomsVal)
x0 = ones(len)
X_val = cat(x0, bedroomsVal, bathroomsVal,sqft_livingVal, dims=2)
Y_val = cat(priceVal,dims=2)

#X, Y of test dataset
len = length(bedroomsTest)
x0 = ones(len)
X_test = cat(x0, bedroomsTest, bathroomsTest,sqft_livingTest, dims=2)
Y_test = cat(priceTest,dims=2)

# Define a function to perform cost gradient
function costFunction(X,Y,B,alp)
   m = length(Y)
#    cost = (sum(((X * B) - Y).^2) + sum(alp.*(B.^2)))/(2*m)
     cost = (sum(((X * B) - Y).^2)/2) +(sum(B.^2))*alp/2
   return cost
end

function gradientDescent(X, Y, B, learningRate, numIterations, alp)
    m = length(Y)
    # do gradient descent for require number of iterations
    for iteration in 1:numIterations
        # Predict with current model B and find loss
        loss = (X * B) - Y
        # Compute Gradients: 
        gradient = ((X' * loss).+(alp.*(B)))/m
        # Perform a descent step in direction oposite to gradient; we want to minimize cost!
        B = B - learningRate * gradient
    end
    return B
end


#APPLYING GRADIENT DESCENT & CROSS VALIDATION TO CALCULATE VALUE
# OF REGULARIZATION COEFFICIENT AT WHICH COST  IS MINIMUM
costm=10^20  # DEFINING RANDOM INITIAL COST VALUE
newB =zeros(4, 1)
lambda=0 
i=10^7
while i>0.001
     learningRate = 0.01
     # # Initial coefficients
     B = zeros(4,1)  
     B1 = gradientDescent(X_train, Y_train, B, learningRate, 1200,i) 
     cost=costFunction(X_val, Y_val, B1, i)
        if(cost<costm)
            costm=cost
            newB=B1
            lambda=i
        end
     i=i/2
end

# Make predictions using the learned model; newB
YPred_train = X_train * newB
YPred_val=X_val * newB
YPred_test = X_test * newB
# print(newB)
# print(i)


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

YPred=[YPred_train;YPred_val;YPred_test]
CSV.write("data/2a.csv",DataFrame(YPred), writeheader=false)
