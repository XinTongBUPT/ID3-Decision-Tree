# -*- coding:utf8 -*-
from node import Node
import math


def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''
  root = Node('', {}, examples, ['Class'])
  return build(root, examples, 'Class')


def prune(node, examples):
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  '''
  
  if node.label == '':
      return node
  temp = getinfogain(node.examples, node.label, 'Class')
  #because most of the maxinfo are above 0.6
  if temp < 0.5:
      tempdict = getdict(node.examples, 'Class')
      maxinfogain = 0.0
      maxattr = ''
      for i in tempdict:
          if tempdict[i] > maxinfogain:
              maxinfogain = tempdict[i]
              maxattr = i
      #change all the values of 'Class' into maxattr
      for item in node.examples:
          item['Class'] = maxattr
      node.label = ''
      for child in node.children:
          node.children[child] = Node('', {}, [], [])
      node.children = {}
  for x in node.children:
      if len(getdict(node.children[x].examples,'Class')) != 1:
        prune(node.children[x],node.children[x].examples)
  return node
  
def test(node, examples):
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''
  right=0
  for example in examples:
      if example['Class']==evaluate(node,example):
         right+=1
  return right/len(examples)
  
  
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''
def evaluate(node, example):
    while(node.children != {}):
        if node.label in example:
            if example[node.label] in node.children.keys():
                node = node.children[example[node.label]]
            else:
                return None
        else:
            break
    for x in node.examples:
        return x['Class']

#get dictionary of every feature
def getdict(example, key):
    dict = {}
    for record in example:
       if record[key] not in dict:
           dict[record[key]]=1
       else:
           dict[record[key]]+=1
    return dict

#get entropy of every feature
def getentropy(example, key):
    dict = getdict(example, key)
    sum = 0
    temp = 0
    for i in dict.keys():
        temp = dict[i]/len(example)
        sum -=(temp*math.log(temp,2))
    return sum

#get information gain of every feature
def getinfogain(example, feature, key):
    #get the dict of key
    resultdict = getdict(example, key)
    #get the dict of feature
    featuredict = getdict(example, feature)
    list = []
    num = {}
    endict = {}
    condientropy = 0.0
    #store(feature,key)in list[]
    for i in featuredict.keys():
        for j in resultdict.keys():
            temp = (i,j)
            list.append(temp)
            #list can only add one data,so I put (i,j) into temp[]
    #store(i,j)& the frequency in num[] accordingly
    for record in example:
        for (i,j) in list:
            if i == record[feature] and j == record[key]:
                if (i,j) not in num.keys():
                    num[(i,j)] = 1
                else:
                    num[(i,j)]+=1
    #get conditional entropy: condientropy  
    for (i,j) in num.keys():
        temp = num[(i,j)]/featuredict[i]
        endict[i] = 0.0;
        endict[i] -= temp*math.log(temp,2)
    for i in endict.keys():
        condientropy += endict[i]*featuredict[i]/len(example)
    #get base entropy
    baseentropy = getentropy(example, key)
    return baseentropy-condientropy

    
#build node of the decision tree
def build(node, example, default):
    #if all attrs are used, return node
    if len(node.usedAttr) == len(example[0].keys()):
        return node
    #if the default('Class') attr of node are all the same values, return node
    if len(getdict(node.examples, default)) == 1:
        return node
    #find the max infogain
    maxinfogain = 0
    attrdict = {}
    maxattr = ''
    for record in node.examples[0]:
        if record in node.usedAttr:
            continue
        attrdict[record] = len(getdict(node.examples, record))
        temp = getinfogain(node.examples, record, default)
        if temp > maxinfogain:
            maxinfogain = temp
            maxattr = record
    #test for pruning
    #print(maxinfogain)
    #if maxinfogain !=0
    if maxinfogain > 0:
        buildChildrenNode(node, maxattr)               
    #if maxinfogain == 0, deal with the situation
    else:
        maxvalue = 0
        for item in attrdict:
            if attrdict[item] > maxvalue:
                maxvalue = attrdict[item]
                maxattr = item
        if maxvalue == 0:
            return False
        else:
            buildChildrenNode(node, maxattr)
    '''
    else:
        maxvalue = 0
        for item in attrdict:
            if attrdict[item] > maxvalue:
                maxvalue = attrdict[item]
                maxattr = item
        if maxvalue == 0:
            return False
        else:
            for record in node.examples:
                record['Class'] = maxattr
    '''
    for i in node.children.values():
        build(i, example, default)
    return node

#build children node
def buildChildrenNode(node, maxattr):
    #update the usedAttr List of node
    node.usedAttr.append(maxattr)
    #update the label of node
    node.label = maxattr
    for item in getdict(node.examples, maxattr).keys():
        newdata = splitdata(node.examples,node.label,item)
        childrenNode = Node('', {}, newdata, node.usedAttr)
        node.children[item] = childrenNode
        #or: node.children.update({item:childrenNode})
    return node

#split data
def splitdata(example, key, value):
    data = []
    for record in example:
        #print(record[key])
        if record[key] == value:
            data.append(record)
    return data
















