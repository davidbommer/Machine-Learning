# Association Analysis
# Association analysis uses machine learning to extract hidden relationships from large datasets. I'll be implementing two of the
# most commonly used algorithms for association rule mining: Apriori and FP-Growth.
# The dataset (`large_retail.txt`) that I'm going to use for this project has been adapted from the 
# [Retail Market Basket Dataset](http://fimi.ua.ac.be/data/retail.pdf). The dataset contains transaction records supplied by an 
# anonymous Belgian retail supermarket store. Each line in the file represents a separate transaction with the item ids separated 
# by space. The dataset has 3000 transaction records and 99 different item ids.


# Part 1 - Apriori Algorithm
# Apriori algorithm is a classical algorithm in data mining. It is used for mining frequent itemsets and relevant 
# association rules. 
# I'll be implementing this algorithm for generating the itemsets that occur more than the min_sup threshold. 
# Based on these frequent itemsets I'll find association rules that have confidence above the min_conf threshold.

# Standard imports
import numpy as np

# Reading the dataset from file
def load_dataset(file_name):
    with open(file_name, 'r') as f:
        content = f.readlines()
        data = [[int(x) for x in line.rstrip().split()] for line in content]
    return data

#take as input the entire dataset and returns a list of all the 1-itemsets.
def create_1_itemsets(dataset):
    c1 = []
    foundSet = set()
    # your code goes here
    for itemset in dataset:
        length = len(foundSet)
        for item in itemset:
            foundSet.add(item)
            if len(foundSet) != length:
                length = len(foundSet)
                c1.append(item)
    c1.sort()
    c1 = [frozenset([x]) for x in c1]
    return c1
#takes as input the candidate itemsets, the dataset, and the minumum support count min_sup, and filters out candidates that 
#don't meet the support threshold.
def itemsetExists(itemset, line):
    for item in itemset:
        if item not in line:
            return False
    return True
def filter_candidates(candidates, dataset, min_sup):
    retlist = []
    support_data = {}
    for itemset in candidates:
        sup = 0
        for dataItemset in dataset:
            if itemsetExists(itemset, dataItemset):
                sup += 1
        if sup >= min_sup:
            retlist.append(itemset)
            support_data[itemset] = sup
    return retlist, support_data
    
#takes in frequent itemsets of size k and generates candidate itemsets of size k + 1.
def comparison(itemset, comp):
    if len(itemset) != len(comp):
        return False
    for index in range(len(itemset) - 1):
        if itemset[index] != comp[index]:
            return False
    return True
def generate_next_itemsets(freq_sets):
    retlist = set()
    # your code goes here
    subsets = set()
    freq_sets.sort()
    for itemset in freq_sets:
        if itemset not in subsets:
            subsets.add(itemset)
            retlist.add(itemset)
        for compItemset in freq_sets:
            itemlist = list(itemset.copy())
            compList = list(compItemset.copy())
            if (itemlist[-1] == compList[-1]):
                continue
            if not comparison(itemlist, compList):
                continue
            items = set(itemlist)
            compItems = set(compList)
            newItemset = items.union(compItems)
            retlist.add(frozenset(newItemset))
        
    return retlist
    
#takes the entire dataset as the input and returns the frequent itemsets that have support count more than min_sup.
def checkLess(itemset, comp):
    if (len(itemset) < len(comp)):
        return True
    elif (len(itemset) == len(comp)):
        for i in range(len(itemset)):
            if comp[i] > itemset[i]:
                return True
        return False
    return False
def sortAlgo(freq):
    n = len(freq)
    for i in range(1, n):
        temp = freq[i]
        j = i
        while(j > 0 and checkLess(temp, freq[j-1])):
            freq[j] = freq[j-1]
            j -= 1
        freq[j] = temp
def apriori_freq_temsets(dataset, minsup):
     # your code goes here
    c1 = create_1_itemsets(dataset)
    candidates, supportDict = filter_candidates(c1, dataset, minsup)
    c2 = set()
    for itemset in candidates:
        for itemset2 in candidates:
            c2.add(itemset.union(itemset2))
    c2, supportDict = filter_candidates(c2, dataset, minsup)
    prevLen = len(c1)
    Len = len(c2)
    while (prevLen != Len):
        c3 = generate_next_itemsets(c2)
        c3, supportDict = filter_candidates(c3, dataset, minsup)
        prevLen = Len
        Len = len(c3)
    sortedList = []
    for frznset in c3:
        itemset = list(frznset)
        sortedList.append(itemset)
    sortAlgo(sortedList)
    return sortedList, supportDict



dataset = load_dataset('large_retail.txt')#change this
freq, supDict = apriori_freq_temsets(dataset, 300)#Change this
frzFreq = []
for itemset in freq:
    freznItem = frozenset(itemset)
    frzFreq.append(freznItem)
for itemset in frzFreq:
    supDict[itemset] = supDict[itemset] / len(dataset)
print("Sup" + "\t" + "Freq Itemset")
for trans, itemset in zip(frzFreq, freq):
    print(str(round(supDict[trans], 2)) + "\t" + str(itemset))
f = open("apriori_itemsets.txt", 'w')
f.write("Sup" + "\t" + "Freq Itemset")
for trans, itemset in zip(frzFreq, freq):
    f.write("\n" + str(round(supDict[trans], 2)) + "\t" + str(itemset))
f.close()
closedSets = []
for index in range(len(frzFreq)-1, -1, -1):
    itemsets = frzFreq[index]
    if len(closedSets) == 0:
        closedSets.append(itemsets)
    isSubset = False
    for closedSet in closedSets:
        if itemsets.issubset(closedSet):
            isSubset = True
            if supDict[itemsets] != supDict[closedSet]:
                closedSets.append(itemsets)
    if not isSubset:
        closedSets.append(itemsets)
closedSets = set(closedSets)
closedSets = list(closedSets)
sortedList = []
for frznset in closedSets:
    itemset = list(frznset)
    sortedList.append(itemset)
sortAlgo(sortedList)
result = []
for itemset in sortedList:
    frzn = frozenset(itemset)
    result.append(frzn)
f = open("apriori_closed_itemsets.txt", 'w')
f.write("Sup" + "\t" + "Freq Itemset")
for trans, itemset in zip(result, sortedList):
    f.write("\n" + str(round(supDict[trans], 2)) + "\t" + str(itemset))
f.close()   





