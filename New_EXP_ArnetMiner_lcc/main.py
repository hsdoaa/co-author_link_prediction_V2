import pandas as pd
import numpy as np
import csv
from pathlib import Path


#print("##########Author_SVM_RF_Default##############")
#print("##########normalization##############")
#print("##########standarization##############")

print("##########No_Feature_scaling##############")


train_path = Path("R_training")
test_path = Path("R_testing")



df = pd.read_csv(train_path/'ad_matrix_training.csv', header=None) #read csv file without header
df.drop(df.columns[0],axis=1,inplace=True)
df.to_csv('ad_matrix_training1.csv',index=False,header=None)

#print(list(df.loc[0]))
print("@@@@@@",len(list(df.loc[0])))
df2=df.loc[0] #1st row (author numbers) in ad_matrix_training.csv
df2.to_csv(train_path/'co_authership_training.csv', index=False, header=['index'])

print(type(df.loc[0]))


#from read_AMiner_Author_50000.py
index=[]
aff=[]
pc_nums =[]
cn_nums=[]
hi_nums=[]
pi_nums=[]
upi_nums=[]
R_intersts=[]

#f = io.open("test", mode="r", encoding="utf-8")  #errors='ignore'           
with open("AMiner_Author.txt", mode="r", encoding="utf-8") as f: #replace AMiner_Author_330000 with AMiner_Author
    for line in f:
        firstWord=line.split(' ', 1)[0]
        rest=line.split(' ', 1)[1:]
        #print(firstWord)
        #print(rest)
        #(firstWord, rest) = line.split(' ', 0)
        if firstWord=='#index': 
              #print(rest[0])   
              rest=int(rest[0])
              #rest=rest[0]  
              index.append(rest)
        elif firstWord=='#a' and rest!=' ':
              #print(len(rest))
              aff.append(rest)   #string should be cleaned from '[ ] ''\n'
        elif firstWord=='#pc' and rest!=' ':
              #print(len(rest))
              #print(type(rest))    
              rest=int(rest[0])
              pc_nums.append(rest)
        elif firstWord=='#cn' and rest!=' ':
              rest=int(rest[0])
              cn_nums.append(rest)
        elif firstWord=='#hi' and rest!=' ':
              rest=int(rest[0])
              hi_nums.append(rest)
        elif firstWord=='#pi' and rest!=' ':
              rest=float(rest[0])
              pi_nums.append(rest)
        elif firstWord=='#upi' and rest!=' ':
              rest=float(rest[0])
              upi_nums.append(rest)      
        elif firstWord=='#t' and rest!=' ':
              #print(";;;",len(rest))
              R_intersts.append(rest) #string should be cleaned from '[ ] ''\n'
              
        else:
            pass


print(len(index))
print("########",index[0])
print(len(pc_nums))
print(len(cn_nums))
print(len(hi_nums))
print(len(pi_nums))
print(len(upi_nums))
print(len(aff))
print(len(R_intersts))

df3=pd.DataFrame()

df3['index']=index
df3['pc_nums']=pc_nums
df3['cn_nums']=cn_nums
df3['hi_nums']=hi_nums
df3['pi_nums']=pi_nums
df3['upi_nums']=upi_nums
df3['aff']=aff
df3['R_intersts']=R_intersts

df3.to_csv('author_dataset.csv', index=False)  #to ignore row index of dataframe

#filter df3 based on the values in co_authership_training.csv

df = pd.read_csv(train_path/'co_authership_training.csv')
df1=df.iloc[:,0]
print(df1.head())
#filter_list=list(df.iloc[:,0])

df2=pd.read_csv('author_dataset.csv')    #ignore index
print(df2.head())
#condition=  df2['index'] in filter_list
#df_new = df[condition]

x=list(set(df.iloc[:,0]).intersection(set(df2.iloc[:,0])))
print("length of intersection list",len(x)) #5625


filtered_data=df2[df2['index'].isin(x)]
filtered_data.to_csv('coauthor_dataset_training.csv',index=False)

#################################
pc_nums_train=[]
cn_nums_train=[]
hi_nums_train=[]
pi_nums_train=[]
upi_nums_train=[]
jaccard_train=[]
dice_train=[]
invlogweighted_train=[]
aa_train=[]
pa_train=[]
cn_train=[]
katz_train=[]
cos_sim_aff_train=[]
cos_sim_RS_train=[]
train_link_label=[]

################################

column1= list(filtered_data['pc_nums'])
for i in column1:
    for j in column1:
        pc_nums_train.append(float(i)+float(j))
print("***",len(pc_nums_train))

column2= list(filtered_data['cn_nums'])
for i in column2:
    for j in column2:
        cn_nums_train.append(float(i)+float(j))
print("***",len(cn_nums_train))

column3= list(filtered_data['hi_nums'])
for i in column3:
    for j in column3:
        hi_nums_train.append(float(i)+float(j))
print("***",len(hi_nums_train))

column4= list(filtered_data['pi_nums'])
for i in column4:
    for j in column4:
        pi_nums_train.append(float(i)+float(j))
print("***",len(pi_nums_train))

column5= list(filtered_data['upi_nums'])
for i in column5:
    for j in column5:
        upi_nums_train.append(float(i)+float(j))
print("***",len(upi_nums_train))

filtered_data['aff'].to_csv('co_author_Affiliations.csv', index=False) 
filtered_data['R_intersts'].to_csv('co_author_R_intersts.csv', index=False) 
##############################################################################

df = pd.read_csv(test_path/'ad_matrix_testing.csv', header=None) #read csv file without header
df.drop(df.columns[0],axis=1,inplace=True)
df.to_csv('ad_matrix_testing1.csv', index=False, header=None)


df22=df.loc[0] #1st row (author numbers) in ad_matrix_training.csv
df22.to_csv(test_path/'co_authership_testing.csv', index=False, header=['index'])

df2=pd.read_csv('author_dataset.csv')    #ignore index
#print(df2.head())

df = pd.read_csv(test_path/'co_authership_testing.csv')

x=list(set(df.iloc[:,0]).intersection(set(df2.iloc[:,0])))
print("length of intersection list",len(x))


filtered_data2=df2[df2['index'].isin(x)]
filtered_data2.to_csv('coauthor_dataset_testing.csv',index=False)

#########################################

pc_nums_test=[]
cn_nums_test=[]
hi_nums_test=[]
pi_nums_test=[]
upi_nums_test=[]

column1= list(filtered_data2['pc_nums'])
for i in column1:
    for j in column1:
        pc_nums_test.append(float(i)+float(j))
print("***",len(pc_nums_test))

column2= list(filtered_data2['cn_nums'])
for i in column2:
    for j in column2:
        cn_nums_test.append(float(i)+float(j))
print("***",len(cn_nums_test))

column3= list(filtered_data2['hi_nums'])
for i in column3:
    for j in column3:
        hi_nums_test.append(float(i)+float(j))
print("***",len(hi_nums_test))

column4= list(filtered_data2['pi_nums'])
for i in column4:
    for j in column4:
        pi_nums_test.append(float(i)+float(j))
print("***",len(pi_nums_test))

column5= list(filtered_data2['upi_nums'])
for i in column5:
    for j in column5:
        upi_nums_test.append(float(i)+float(j))
print("***",len(upi_nums_test))

filtered_data2['aff'].to_csv('co_author_Affiliations_test.csv', index=False) 
filtered_data2['R_intersts'].to_csv('co_author_R_intersts_test.csv', index=False) 
#######################################################
train_dataset=pd.DataFrame()

train_dataset['pc_nums']=pc_nums_train
train_dataset['cn_nums']=cn_nums_train
train_dataset['hi_nums']=hi_nums_train
train_dataset['pi_nums']=pi_nums_train
train_dataset['upi_nums']=upi_nums_train



with open(train_path/"jaccard_train.csv") as f:
     reader = csv.reader(f)   
     next(reader)      # to ignore first row 
     for row in reader:
        jaccard_train.append(row[1])
print(len(jaccard_train))
train_dataset['jaccard']=pd.Series(jaccard_train) #to avoid error at https://stackoverflow.com/questions/42382263/valueerror-length-of-values-does-not-match-length-of-index-pandas-dataframe-u


with open(train_path/"dice_train.csv") as f:
     reader = csv.reader(f)   
     next(reader)      # to ignore first row 
     for row in reader:
        dice_train.append(row[1])
print(len(dice_train))
train_dataset['dice']=pd.Series(dice_train)

with open(train_path/"invlogweighted_train.csv") as f:
     reader = csv.reader(f)   
     next(reader)      # to ignore first row 
     for row in reader:
        invlogweighted_train.append(row[1])
print("]]]",len(invlogweighted_train))
train_dataset['invlogweighted']=pd.Series(invlogweighted_train)

with open(train_path/"aa_train.csv") as f:
     reader = csv.reader(f)   
     next(reader)      # to ignore first row 
     for row in reader:
        aa_train.append(row[1])
print("]]]",len(aa_train))
train_dataset['aa']=pd.Series(aa_train)

with open(train_path/"pa_train.csv") as f:
     reader = csv.reader(f)   
     next(reader)      # to ignore first row 
     for row in reader:
        pa_train.append(row[1])
print("]]]",len(pa_train))
train_dataset['pa']=pd.Series(pa_train)


with open(train_path/"cn_train.csv") as f:
     reader = csv.reader(f)   
     next(reader)      # to ignore first row 
     for row in reader:
        cn_train.append(row[1])
print("]]]",len(cn_train))
train_dataset['cn']=pd.Series(cn_train)

with open(train_path/"katz_train.csv") as f:
     reader = csv.reader(f)   
     next(reader)      # to ignore first row 
     for row in reader:
        katz_train.append(row[1])
print("]]]",len(katz_train))
train_dataset['katz']=pd.Series(katz_train)

with open(train_path/"cos_sim_aff_train.csv") as f:
     reader = csv.reader(f)   
     next(reader)      # to ignore first row 
     for row in reader:
        for i in row:
           cos_sim_aff_train.append(i)
print(";;;",len(cos_sim_aff_train))
train_dataset['cos_sim_aff']=pd.Series(cos_sim_aff_train)
with open(train_path/"cos_sim_RS_train.csv") as f:
     reader = csv.reader(f)   
     next(reader)      # to ignore first row 
     for row in reader:
        for i in row:
           cos_sim_RS_train.append(i)
print("----",len(cos_sim_RS_train))
train_dataset['cos_sim_RS']=pd.Series(cos_sim_RS_train)


with open("ad_matrix_training1.csv") as f:
     reader = csv.reader(f)   
     next(reader)      # to ignore first row 
     for row in reader:
        for i in row:
           train_link_label.append(i)
print("$$$",len(train_link_label))
train_dataset['label']=pd.Series(train_link_label)


print(train_dataset.shape)
print(train_dataset.head())
features=train_dataset.columns
print("train_features=",features)
#X_train = df4[features]
#print(X_train.shape)
#print(X_train.head())
#df4.to_csv('link_taining_dataset.csv',index=False, header=['jaccard','dice','invlogweighted','cos_sim_aff','cos_sim_RS','label'])

#################################
test_dataset=pd.DataFrame()
test_dataset['pc_nums']=pc_nums_test
test_dataset['cn_nums']=cn_nums_test
test_dataset['hi_nums']=hi_nums_test
test_dataset['pi_nums']=pi_nums_test
test_dataset['upi_nums']=upi_nums_test

jaccard_test=[]
dice_test=[]
invlogweighted_test=[]
aa_test=[]
pa_test=[]
cn_test=[]
katz_test=[]

cos_sim_aff_test=[]
cos_sim_RS_test=[]
test_link_label=[]

with open(test_path/"jaccard_test.csv") as f:
     reader = csv.reader(f)   
     next(reader)      # to ignore first row 
     for row in reader:
        jaccard_test.append(row[1])
print(len(jaccard_test))
test_dataset['jaccard']=pd.Series(jaccard_test)

with open(test_path/"dice_test.csv") as f:
     reader = csv.reader(f)   
     next(reader)      # to ignore first row 
     for row in reader:
        dice_test.append(row[1])
print(len(dice_train))
test_dataset['dice']=pd.Series(dice_test)

with open(test_path/"invlogweighted_test.csv") as f:
     reader = csv.reader(f)   
     next(reader)      # to ignore first row 
     for row in reader:
        invlogweighted_test.append(row[1])
print("]]]",len(invlogweighted_test))
test_dataset['invlogweighted']=pd.Series(invlogweighted_test)

with open(test_path/"aa_test.csv") as f:
     reader = csv.reader(f)   
     next(reader)      # to ignore first row 
     for row in reader:
        aa_test.append(row[1])
print("]]]",len(aa_test))
test_dataset['aa']=pd.Series(aa_test)

with open(test_path/"pa_test.csv") as f:
     reader = csv.reader(f)   
     next(reader)      # to ignore first row 
     for row in reader:
        pa_test.append(row[1])
print("]]]",len(pa_test))
test_dataset['pa']=pd.Series(pa_test)


with open(test_path/"cn_test.csv") as f:
     reader = csv.reader(f)   
     next(reader)      # to ignore first row 
     for row in reader:
        cn_test.append(row[1])
print("]]]",len(cn_test))
test_dataset['cn']=pd.Series(cn_test)

with open(test_path/"katz_test.csv") as f:
     reader = csv.reader(f)   
     next(reader)      # to ignore first row 
     for row in reader:
        katz_test.append(row[1])
print("]]]",len(katz_test))
test_dataset['katz']=pd.Series(katz_test)

with open(test_path/"cos_sim_aff_test.csv") as f:
     reader = csv.reader(f)   
     next(reader)      # to ignore first row 
     for row in reader:
        for i in row:
           cos_sim_aff_test.append(i)
print("]]]",len(cos_sim_aff_test))
test_dataset['cos_sim_aff']=pd.Series(cos_sim_aff_test)
with open(test_path/"cos_sim_RS_test.csv") as f:
     reader = csv.reader(f)   
     next(reader)      # to ignore first row 
     for row in reader:
        for i in row:
           cos_sim_RS_test.append(i)
print("]]]",len(cos_sim_RS_test))
test_dataset['cos_sim_RS']=pd.Series(cos_sim_RS_test)

with open("ad_matrix_testing1.csv") as f:
     reader = csv.reader(f)   
     next(reader)      # to ignore first row 
     for row in reader:
        for i in row:
           test_link_label.append(i)
print("]]]",len(test_link_label))
test_dataset['label']=pd.Series(test_link_label)
print(test_dataset.shape)
print(test_dataset.head())
#features2=test_dataset.columns
#print("test_features=",features2)


print("==============")
print(train_dataset.info())
print(test_dataset.info())
print("==============")

print(train_dataset.dtypes)
print(test_dataset.dtypes)

train_dataset['jaccard'] = train_dataset['jaccard'].astype(str).astype(float)
train_dataset['dice'] = train_dataset['dice'].astype(str).astype(float)
train_dataset['invlogweighted'] = train_dataset['invlogweighted'].astype(str).astype(float)
train_dataset['cos_sim_aff'] = train_dataset['cos_sim_aff'].astype(str).astype(float)
train_dataset['cos_sim_RS'] =train_dataset['cos_sim_RS'].astype(str).astype(float)
train_dataset['label'] =train_dataset['label'].astype(str).astype(int)
print(train_dataset.dtypes)

test_dataset['jaccard'] = test_dataset['jaccard'].astype(str).astype(float)
test_dataset['dice'] = test_dataset['dice'].astype(str).astype(float)
test_dataset['invlogweighted'] = test_dataset['invlogweighted'].astype(str).astype(float)
test_dataset['cos_sim_aff'] = test_dataset['cos_sim_aff'].astype(str).astype(float)
test_dataset['cos_sim_RS'] =test_dataset['cos_sim_RS'].astype(str).astype(float)
test_dataset['label'] =test_dataset['label'].astype(str).astype(int)
print(test_dataset.dtypes)


print("==============")
print(train_dataset.info())
print(test_dataset.info())

##########################################################################

from sklearn.neural_network import MLPClassifier
#classifier = MLPClassifier()

from sklearn.linear_model import LogisticRegression
#classifier=LogisticRegression()

from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()

from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier()

from sklearn import svm
classifier =svm.SVC() #classifier =svm.SVC(gamma='scale',C=1,probability=True)

from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier()#classifier = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=0)

print("CCCCCC", classifier)

from sklearn.preprocessing import MinMaxScaler #For feature normalization
from sklearn.preprocessing import StandardScaler # For feature standarization

#scaler = MinMaxScaler()
#scaler = StandardScaler()

#print("SSSSSS",scaler )

#train_existing_links=train_dataset[train_dataset['label']!=0]

train_existing_links=train_dataset.query('label!=0')
#train_missing_links=train_dataset[train_dataset['label']==0]
train_missing_links=train_dataset.query('label==0')
train_existing_links['label'] =1

print(train_existing_links.shape)
print(train_existing_links.head())
print(train_missing_links.shape)
print(train_missing_links.head())

# Remove duplicates
#train_existing_links = train_existing_links.drop_duplicates()
#train_missing_links = train_missing_links.drop_duplicates()
# Down sample negative examples
train_missing_links = train_missing_links.sample(n=len(train_existing_links), replace=True) # sampling without replacement-try replace=False

# Create DataFrame from positive and negative examples
training_df = train_missing_links.append(train_existing_links, ignore_index=True)
training_df['label'] = training_df['label'].astype('category')
print(training_df.head())

test_existing_links=test_dataset[test_dataset['label']!=0]
test_missing_links=test_dataset[test_dataset['label']==0]
test_existing_links['label'] =1

print(test_existing_links.shape)
print(test_missing_links.shape)

# Remove duplicates
#test_existing_links = test_existing_links.drop_duplicates()
#test_missing_links = test_missing_links.drop_duplicates()
# Down sample negative examples
test_missing_links = test_missing_links.sample(n=len(test_existing_links), replace=True) # sampling with replacement

# Create DataFrame from positive and negative examples
testing_df = test_missing_links.append(test_existing_links, ignore_index=True)
testing_df['label'] = testing_df['label'].astype('category')
print(testing_df.head())



############################################################################################################
#features=['pc_nums', 'cn_nums', 'hi_nums', 'pi_nums', 'upi_nums','cos_sim_aff','cos_sim_RS','jaccard','dice', 'invlogweighted']#ICMLC2019
#features=['pc_nums', 'cn_nums', 'hi_nums', 'pi_nums', 'upi_nums','cos_sim_aff','cos_sim_RS']
#features=['jaccard','dice', 'invlogweighted', 'aa', 'pa', 'cn', 'katz']
#features=['pc_nums', 'cn_nums', 'hi_nums', 'pi_nums', 'upi_nums', 'jaccard','dice', 'invlogweighted', 'aa', 'pa', 'cn', 'katz', 'cos_sim_aff','cos_sim_RS']
##########################################################################################################################################################

################Feature importance#############################################
#features=['pc_nums', 'cn_nums', 'hi_nums', 'pi_nums', 'upi_nums']
features=['cos_sim_aff']
#features=['cos_sim_RS']
#features=['jaccard','dice', 'invlogweighted']
##############################################





#columns = []
# Train RF Classifer
X_train = training_df[features]
#X_train= scaler.fit_transform(X_train) #scale training features
y_train = training_df['label']


clf = classifier.fit(X_train,y_train)

#Predict the response for test dataset
X_test = testing_df[features]
#X_test= scaler.fit_transform(X_test)  #scale testing features
y_test = testing_df["label"]
y_pred = classifier.predict(X_test)


# Evaluate the model: Model Accuracy, how often is the classifier correct
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report #for classifier evaluation
from sklearn.metrics import roc_auc_score # for printing AUC


#print(classifier)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)
 
print(classification_report(y_test, y_pred))
auc=roc_auc_score(y_test,y_pred)
auc = float("{0:.3f}".format(auc))
print("AUC=",auc)

print(features)
print(classifier)
#print(scaler)



