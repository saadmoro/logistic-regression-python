import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Load training data features and response
training_data = pd.read_csv('pa2_train_X.csv')
y = pd.read_csv('pa2_train_y.csv')

validation_data = pd.read_csv('pa2_dev_X.csv')
validation_y = pd.read_csv('pa2_dev_y.csv')

#fetch variable names
names = training_data.columns.values.tolist()

#Table creation + formatting
def make_table(titles, data):
    # find longest string
    longest_lengths = []
    for item in titles:
        longest = 0
        if len(item) > longest:
            longest = len(item)
        for item in pd.DataFrame(data).T.values.tolist()[titles.index(item)]:
            if len(str(item)) > longest:
                longest = len(str(item))
        longest_lengths.append(longest + 2)
    top = "+"
    for item in titles:
        top += longest_lengths[titles.index(item)] * '-' + '+'
    print(top)
    string = "| "
    for item in titles:
        padding = longest_lengths[titles.index(item)] - len(item) - 1
        string += str(item) + " " * padding + "| "
    print(string)
    print(top)
    string = "| "
    for col in data:
        string = "| "
        for item in col:
            padding = longest_lengths[col.index(item)] - len(str(item)) - 1
            string += str(item) + " " * padding + "| "
        print(string)
    print(top)

#Function to perform min-max normalization
def mmnormalize(vec):
    v = vec
    vec = (v - v.min())/(v.max() - v.min())
    return vec

#Normalize the numeric and ordinal features
training_data['Age'] = mmnormalize(training_data['Age'])
training_data['Annual_Premium'] = mmnormalize(training_data['Annual_Premium'])
training_data['Vintage'] = mmnormalize(training_data['Vintage'])

validation_data['Age'] = mmnormalize(validation_data['Age'])
validation_data['Annual_Premium'] = mmnormalize(validation_data['Annual_Premium'])
validation_data['Vintage'] = mmnormalize(validation_data['Vintage'])

#Array initialization
training_data = np.c_[training_data]
y = np.c_[y]
y = y.flatten() #y doesn't load as a 1-D array; make it so

validation_data = np.c_[validation_data]
validation_y = np.c_[validation_y]
validation_y = validation_y.flatten()


#Loss calculation - unused
def loss(w, X, y, reg):
    N = len(y)
    Xw = np.dot(X, w)
    sig = sigmoid(Xw)
    log1 = np.log(sig)
    log2 = np.log(1-sig)
    avg = (1/N)*np.sum(-y * log1 - (1-y)*log2)
    regpart = reg*np.dot(w, w)
    return avg + regpart

#Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Logistic Regression with L2 (ridge) regularization
def logisticRegression_L2(X, y, rate, reg):
    N,d = np.shape(X)
    w = np.zeros(d)
    #loss_history = []

    for i in range(0,10000): #10000 max iterations
        w_old = w

        sig = sigmoid(np.dot(X, w))
        sums = np.dot((y - sig), X)
        w = w + (rate/N)*sums #normal gradient without L2 norm
        w = w - (rate*reg)*w #L2 norm contribution
        #loss_history.append(loss(w, X, y, reg))

        if np.linalg.norm(w - w_old) < 0.0001: #maybe improve on stopping condition?
            print("Iterations to convergence:")
            print(i)
            break
    
    return w

#Based on a weight vector w and data matrix X, classify each row as a 1 or 0
#(using wtx=0 as a linear decision boundary)
def decisionBoundary(X, w):
    boundary = np.dot(X,w)
    classification = np.where(boundary > 0, 1, 0)
    return classification

#Determine accuracy by counting discrepancies between vectors
def determineAccuracy(y, yhat):
    diff = y - yhat #Difference between estimate and actual
    boolDiff = np.where(diff == 0, 1, 0) #diff of 0 means accurate estimate
    return np.mean(boolDiff)

#Determine the accuracy of a weight vector w, compare with actual y
def testWeights(X, y, w):
    fittedClasses = decisionBoundary(X, w)
    successRate = determineAccuracy(y, fittedClasses)
    return successRate

#Plot the different regularization parameters we want to test
def testRegularization(trainX, trainY, validX, validY, rate):
    trainingAccuracy = np.zeros(6)
    validationAccuracy = np.zeros(6)

    w_collection = []

    for i in range(0,6):
        reg = 10**-i

        print("\nTraining L2 regularized logistic regression for lambda=", reg)
        w = logisticRegression_L2(trainX, trainY, rate, reg)
        w_collection.append(w)
        print("Zeros in w for lambda=", reg, ": ", len(w) - np.count_nonzero(np.abs(w)))
        trainingAccuracy[i] = tacc = 100*testWeights(trainX, trainY, w)#convert to %
        validationAccuracy[i] = vacc = 100*testWeights(validX, validY, w)
        print("Training accuracy for lambda =", reg, ": ", tacc)
        print("Validation accuracy for lambda =", reg, ": ", vacc)

    fig,ax = plt.subplots()

    ind = np.arange(6)
    width = 0.3

    ax.bar(ind - width/2, trainingAccuracy, width, label = 'Training')
    ax.bar(ind + width/2, validationAccuracy, width, label = 'Validation')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Ridge logistic regression accuracies')
    ax.set_ylim([0,100])
    ax.set_xticks(ind)
    ax.set_xlabel('Regularization parameter')
    labels = ['10e0', '10e-1', '10e-2', '10e-3', '10e-4', '10e-5']
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    #display plot
    plt.show()

    return trainingAccuracy, validationAccuracy, w_collection


t, v, w_col = testRegularization(training_data, y, validation_data, validation_y, 0.1)

#retrieve most accurate (validation) weight vector
print('Most accurate weights achieved at regularization parameter: 10e-', np.argmax(v))
w_max = w_col[np.argmax(validation_data)]


combined = []
for i in range(len(names)):
    combined.append([names[i], w_max[i]])

combined.sort(key = lambda x: 1-abs(x[1]))

#Select top 5 most heavily weighted features
combined = combined[:5]

make_table(['Feature', 'Weight'], combined)

