import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


def splitDataset(dataset,splitRatio):
    length = int(len(dataset))
    trainSize = int(splitRatio * length)
    trainSize2=int(trainSize)
    trainSet = dataset.iloc[0:trainSize2 , :].values
    testSet = dataset.iloc[trainSize2:length , :].values
    return [trainSet , testSet]
    

def sortTrainSet(trainSet):
    class1 = []
    for i in range(len(trainSet)):
        class1.append(trainSet[i])
    class1.sort(key=lambda x:x[312])
    return class1
    
def serrateClases(trainSet):
    class1 = []
    for i in range(len(trainSet)):
        if trainSet[i][312] == 1:
            pass
        else:
            print('we stoped at index ' + str(i))
            break
    for j in range(i):
        class1.append(trainSet[j])
    return class1

def calcclassprobabilty (class1 , trainset):
    probabilty = (len(class1) / len(trainset))
    return probabilty

def calculateRowMean(trainItem):
    sum = 0
    for i in range(309):
        sum =sum + trainItem[i]
    mean = sum / 309
    return mean

  
def calculateRowpower(trainItem):
    sum = 0
    for i in range(309):
        sum = sum + ((trainItem[i]) ** 4) 
    mean_of_power= sum / 309
    return mean_of_power

def calculateRowenergy(trainItem):
    sum = 0
    for i in range(309):
        sum = sum + ((trainItem[i]) ** 2)     
    mean_of_energy= sum / 309
    return mean_of_energy

def calculateRowcurvelenght(trainItem):
    sum = 0
    for i in range(1,309):
        sum = sum + (trainItem[i]) * (trainItem[i-1])
    mean_of_curvelenght= sum / 309
    return mean_of_curvelenght

def calculateRownonlinearenergy(trainItem):
    sum = 0
    i=2
    for i in range(2,309):
        sum =sum + ((-trainItem[i])*(trainItem[i-2]))+((trainItem[i-1])**2)
    mean_of_nonlinearenergy= sum / 309
    return mean_of_nonlinearenergy



def calculateMM_MP_ME_MC_MN(trainClass):
    meanSum = 0
    powersum = 0
    energysum = 0
    meanOfmeans = 0
    curvelenghtsum = 0
    nonlinearenergysum = 0
    meanOfpowers = 0
    meanofcurvelenght=0
    meanofnonlinearenergy=0
    for i in range(len(trainClass)):
    #for i in range(1):
        itemMean =  calculateRowMean(trainClass[i])
        meanSum =meanSum +  itemMean
        itempower = calculateRowpower(trainClass[i])
        powersum =powersum + itempower
        itemenergy = calculateRowenergy(trainClass[i])
        energysum = energysum + itemenergy 
        itemcurvelenght = calculateRowcurvelenght(trainClass[i])
        curvelenghtsum =curvelenghtsum + itemcurvelenght
        itemnonlinearenergy = calculateRownonlinearenergy(trainClass[i])
        nonlinearenergysum = nonlinearenergysum + itemnonlinearenergy
    meanOfmeans = meanSum / len(trainClass)
    meanOfpowers = powersum / len(trainClass)
    meanOfenergy = energysum / len(trainClass)
    meanofcurvelenght = curvelenghtsum / len(trainClass)
    meanofnonlinearenergy = nonlinearenergysum / len(trainClass)
    return[meanOfmeans , meanOfpowers, meanOfenergy , meanofcurvelenght , meanofnonlinearenergy]


def calculate_stdev_MPE(trainClass):
    num_stdev_m = 0
    num_stdev_p = 0
    num_stdev_E = 0
    num_stdev_C = 0
    num_stdev_N = 0
    meanOfMeans , meanOfpowers, meanOfenergy,meanOfcurvelenght , meanOfnonlinearenergy = calculateMM_MP_ME_MC_MN(trainClass)
    for i in range(len(trainClass)):
        stdev_m = pow(calculateRowMean(trainClass[i])-meanOfMeans,2)
        num_stdev_m = num_stdev_m + stdev_m
        stdev_p = pow(calculateRowpower(trainClass[i])-meanOfpowers,2)
        num_stdev_p = num_stdev_p + stdev_p
        stdev_E = pow(calculateRowenergy(trainClass[i])-meanOfenergy,2)
        num_stdev_E = num_stdev_E + stdev_E
        stdev_C = pow(calculateRowcurvelenght(trainClass[i])-meanOfcurvelenght,2)
        num_stdev_C = num_stdev_C + stdev_C
        stdev_N = pow(calculateRownonlinearenergy(trainClass[i])-meanOfnonlinearenergy,2)
        num_stdev_N = num_stdev_N + stdev_N
    stdevofMeans = math.sqrt (num_stdev_m / (len(trainClass)-1))
    stdevofpowers = math.sqrt (num_stdev_p / (len(trainClass)-1))
    stdevofenergy = math.sqrt (num_stdev_E / (len(trainClass)-1))
    stdevofcurve = math.sqrt (num_stdev_C / (len(trainClass)-1))
    stdevofnonlinearty = math.sqrt (num_stdev_N / (len(trainClass)-1))
    return [stdevofMeans , stdevofpowers , stdevofenergy , stdevofcurve,  stdevofnonlinearty ]


def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def checkClass(pridictions):
    if pridictions > 0.5 :
        return 1 
    else:
        return 2

def pridections(testSet):
    testMean = []
    testPower = []
    testEnergy = []
    testCurve = []
    testNonLinearity = []
    for i in range(len(testSet)):
        itemMean = calculateRowMean(testSet[i])
        testMean.append(itemMean)
        itemPower = calculateRowpower(testSet[i])
        testPower.append(itemPower)
        itemEnergy = calculateRowenergy(testSet[i])
        testEnergy.append(itemEnergy)
        itemCurve = calculateRowcurvelenght(testSet[i])
        testCurve.append(itemCurve)
        itemNonLinearity = calculateRownonlinearenergy(testSet[i])
        testNonLinearity.append(itemNonLinearity)
    return [testMean,testPower,testEnergy,testCurve,testNonLinearity]
    
    
    
def getFeaturesPridictions(class1,testSet):
    Mean_p = []
    Power_p = []
    Energy_p = []
    Curve_p = []
    testNonLinearity = []
    meanOfMeans , meanOfpowers, meanOfenergy , meanOfcurvelenght , meanOfnonlinearenergy = calculateMM_MP_ME_MC_MN(class1)
    stdevofMeans , stdevofpowers , stdevofenergy , stdevofcurve,  stdevofnonlinearty = calculate_stdev_MPE(class1)
    testMean,testPower,testEnergy,testCurve,testNonLinearity = pridections(testSet)
    for i in range(len(testSet)-1):
        item_mean_p = calculateProbability(testMean[i],meanOfMeans,stdevofMeans)
        Mean_p.append(item_mean_p)
        item_power_p = calculateProbability(testPower[i],meanOfpowers,stdevofpowers)
        Power_p.append(item_power_p)
        item_energy_p = calculateProbability(testEnergy[i],meanOfenergy,stdevofenergy)
        Energy_p.append(item_energy_p)
        item_curve_p = calculateProbability(testCurve[i],meanOfcurvelenght,stdevofcurve)
        Curve_p.append(item_curve_p)
        item_NonLinearity_p = calculateProbability(testNonLinearity[i],meanOfnonlinearenergy,stdevofnonlinearty)
        testNonLinearity.append(item_NonLinearity_p)
        
    return [Mean_p,Power_p,Energy_p,Curve_p,testNonLinearity]


def getPridictions(class1 , trainset,testSet):
    prediction = []
    Mean_p,Power_p,Energy_p,Curve_p,testNonLinearity = getFeaturesPridictions(class1,testSet)
    classProbability = calcclassprobabilty(class1,trainset)
    print('this is items probability')
    for i in range(len(testSet)-1):
        itemProbability = Mean_p[i]*Power_p[i]*Energy_p[i]*Curve_p[i]*testNonLinearity[i]*classProbability*100
    
        itemClass = checkClass(itemProbability)
        prediction.append(itemClass) 
    print(prediction)
    return prediction



def getAccuracy(prediction,testSet):
    correct = 0
    for i in range(len(prediction)):
        if prediction[i] == testSet[i][-1]:
            correct = correct + 1
           
            
    accuracy = (correct/len(prediction))*100
    return accuracy

def formatTrainSet(testSet):
    tList = []
    for i in range(len(testSet)):
        tList.append(testSet[i][0:310])
    return tList
 
    
        
        
def main(fileName,splitRatio):
    dataset = pd.read_excel('LSVT_voice_rehabilitation.xlsx')
    splitRatio = 0.8
    trainSet , testSet = splitDataset(dataset,splitRatio)
    sortedTrainSet = sortTrainSet(trainSet)
    class1 = serrateClases(sortedTrainSet)
    tList = formatTrainSet(testSet)
    Mean_p,Power_p,Energy_p,Curve_p,testNonLinearity = getFeaturesPridictions(class1,tList)
    prediction = getPridictions(class1,trainSet,tList)
    accuracy = getAccuracy(prediction,testSet)
    print('Accuracy Equal '+str(accuracy))

main('LSVT_voice_rehabilitation.xlsx',0.8)


