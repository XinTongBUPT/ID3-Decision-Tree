import ID3, parse, random
import pandas as pd
def testID3AndEvaluate():
  data = [dict(a=1, b=0, Class=1), dict(a=1, b=1, Class=1)]
  tree = ID3.ID3(data, 0)
  if tree != None:
    ans = ID3.evaluate(tree, dict(a=1, b=0))
    if ans != 1:
      print("ID3 test failed.")
    else:
      print("ID3 test succeeded.")
  else:
    print("ID3 test failed -- no tree returned")

def testPruning():
  # data = [dict(a=1, b=1, c=1, Class=0), dict(a=1, b=0, c=0, Class=0), dict(a=0, b=1, c=0, Class=1), dict(a=0, b=0, c=0, Class=1), dict(a=0, b=0, c=1, Class=0)]
  # validationData = [dict(a=0, b=0, c=1, Class=1)]
  data = [dict(a=0, b=1, c=1, d=0, Class=1), dict(a=0, b=0, c=1, d=0, Class=0), dict(a=0, b=1, c=0, d=0, Class=1), dict(a=1, b=0, c=1, d=0, Class=0), dict(a=1, b=1, c=0, d=0, Class=0), dict(a=1, b=1, c=0, d=1, Class=0), dict(a=1, b=1, c=1, d=0, Class=0)]
  validationData = [dict(a=0, b=0, c=1, d=0, Class=1), dict(a=1, b=1, c=1, d=1, Class = 0)]
  tree = ID3.ID3(data, 0)
  ID3.prune(tree, validationData)
  if tree != None:
    ans = ID3.evaluate(tree, dict(a=0, b=0, c=1, d=0))
    print(ans)
    if ans != 1:
      print("pruning test failed.")
    else:
      print("pruning test succeeded.")
  else:
    print("pruning test failed -- no tree returned.")


def testID3AndTest():
  trainData = [dict(a=1, b=0, c=0, Class=1), dict(a=1, b=1, c=0, Class=1), 
  dict(a=0, b=0, c=0, Class=0), dict(a=0, b=1, c=0, Class=1)]
  testData = [dict(a=1, b=0, c=1, Class=1), dict(a=1, b=1, c=1, Class=1), 
  dict(a=0, b=0, c=1, Class=0), dict(a=0, b=1, c=1, Class=0)]
  tree = ID3.ID3(trainData, 0)
  fails = 0
  if tree != None:
    acc = ID3.test(tree, trainData)
    if acc == 1.0:
      print("testing on train data succeeded.")
    else:
      print("testing on train data failed.")
      fails = fails + 1
    acc = ID3.test(tree, testData)
    if acc == 0.75:
      print("testing on test data succeeded.")
    else:
      print("testing on test data failed.")
      fails = fails + 1
    if fails > 0:
      print("Failures: ", fails)
    else:
      print("testID3AndTest succeeded.")
  else:
    print("testID3andTest failed -- no tree returned.")    

def testPruningOnHouseData(inFile):
  withPruning = []
  withoutPruning = []
  data = parse.parse(inFile)
  for i in range(100):
    random.shuffle(data)
    train = data[:len(data)//2]
    valid = data[len(data)//2:3*len(data)//4]
    test = data[3*len(data)//4:]
    train=train[0:30]
    tree = ID3.ID3(train, 'democrat')
    acc = ID3.test(tree, train)
    print("training accuracy: ",acc)
    acc = ID3.test(tree, valid)
    print("validation accuracy: ",acc)
    acc = ID3.test(tree, test)
    print("test accuracy: ",acc)
    ID3.prune(tree, valid)
    acc = ID3.test(tree, train)
    print("pruned tree train accuracy: ",acc)
    acc = ID3.test(tree, valid)
    print("pruned tree validation accuracy: ",acc)
    acc = ID3.test(tree, test)
    print("pruned tree test accuracy: ",acc)
    withPruning.append(acc)
    tree = ID3.ID3(train+valid, 'democrat')
    acc = ID3.test(tree, test)
    print("no pruning test accuracy: ",acc)
    withoutPruning.append(acc)
  print(withPruning)
  print(withoutPruning)
  print("average with pruning",sum(withPruning)/len(withPruning)," without: ",sum(withoutPruning)/len(withoutPruning))
  
# inFile - string location of the house data file
def testPruningOnHouseData1(inFile):
  withPruning = []
  withoutPruning = []
  data = parse.parse(inFile)
  array = []
  i = 30
  acc = 0.0
  while i <301:
      array.append(i)
      i += 1
  accwithPruning = []
  accwithoutPruning = []
  for item in array:
      withPruning = []
      withoutPruning = []
      for i in range(20):
          random.shuffle(data)
          train = data[:len(data)//2]
          valid = data[len(data)//2:3*len(data)//4]
          test = data[3*len(data)//4:]
          train=train[0:item]
          tree = ID3.ID3(train, 'democrat')
          ID3.prune(tree, valid)
          acc = ID3.test(tree, train)
          withPruning.append(acc)
          tree=ID3.prune(tree, test+valid)
          acc=ID3.test(tree,valid+test)
          tree = ID3.ID3(train+valid, 'democrat')
          acc = ID3.test(tree, test)
          withoutPruning.append(acc)
      accwithPruning.append(sum(withPruning)/len(withPruning))
      accwithoutPruning.append(sum(withoutPruning)/len(withoutPruning))

  print('accwithPruning')
  print(accwithPruning)
  x=pd.DataFrame(accwithPruning)
  x.to_csv('x.csv')
  y=pd.DataFrame(accwithoutPruning)
  y.to_csv('y.csv')
  print('accwithoutPruning')
  print(accwithoutPruning)
    
def testPruningOnHouseData2(inFile):
  data = parse.parse(inFile)
  array = []
  i = 30
  acc = 0.0
  while i <301:
      array.append(i)
      i += 1
  accwithPruning = []
  accwithoutPruning = []
  for item in array:
      withPruning = []
      withoutPruning = []
      for i in range(100):
          random.shuffle(data)
          train = data[:len(data)//2]
          valid = data[len(data)//2:3*len(data)//4]
          test = data[3*len(data)//4:]
          train=train[0:item]
          tree = ID3.ID3(train, 'democrat')
          acc=ID3.test(tree,test)
          print('accwithoutprune')
          print(acc)
          withoutPruning.append(acc)
          tree=ID3.prune(tree, train+valid)
          acc=ID3.test(tree,test)
          print('accwithprune')
          print(acc)
          withPruning.append(acc)
      #print("average with pruning",sum(withPruning)/len(withPruning)," without: ",sum(withoutPruning)/len(withoutPruning))
      accwithPruning.append(sum(withPruning)/len(withPruning))
      accwithoutPruning.append(sum(withoutPruning)/len(withoutPruning))
  print('accwithPruning')
  print(accwithPruning)
  x=pd.DataFrame(accwithPruning)
  x.to_csv('x.csv')
  y=pd.DataFrame(accwithoutPruning)
  y.to_csv('y.csv')
  print('accwithoutPruning')
  print(accwithoutPruning)
  #print(withPruning)
  #print(withoutPruning)
  

#testID3AndEvaluate()
#testPruning()
