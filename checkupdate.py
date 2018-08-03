from firebase import firebase
import sqlite3
firebase = firebase.FirebaseApplication('https://dogwood-terra-184417.firebaseio.com/', None)
conn = sqlite3.connect('C:\Users\MOJI\Desktop\project\Firebase\python\Test_PJ2.db')
cursor = conn.execute("SELECT ID,Name,day,month,year,hour,minute from ActionName")
result = firebase.get('/action/data/',None)
result1 = firebase.get('/object/data/',None)
n = []
Na=[]
nM = []
NaM=[]
for row in cursor:
    NaM.append(row[1])
cursor.close()
for i in result :
    name=firebase.get('/action/data/'+i,'action_name')
    nM.append(name)
nMocheck = set(nM).difference(NaM)
nMocheck = list(nMocheck)
ln3=len(nMocheck)   
nMocheck2 = set(NaM).difference(nM)
nMocheck2 = list(nMocheck2)
ln4=len(nMocheck2)
cursor = conn.execute("SELECT ID,Name,day,month,year,hour,minute from call_Detect")
for row in cursor:
    Na.append(row[1])
cursor.close()   

for i in result1 :
    name=firebase.get('/object/data/'+i,'object_Name')
    n.append(name)
 
nobcheck = set(n).difference(Na)
nobcheck = list(nobcheck)
ln=len(nobcheck)
nobcheck2 = set(Na).difference(n)
nobcheck2 = list(nobcheck2)
ln2=len(nobcheck2)
picture=ln3+ln4 
motor=ln+ln2
#print motor,picture
if(motor!=0 and picture!=0):
    result2 = firebase.get('/updatecheck/data/',None)
    if result2!=None :
        result2 = firebase.delete('/updatecheck/data/',None)     
    firebase.post('/updatecheck/data', {'check':"Object information does not match and Action information does not match.Please press update button."})
elif(motor!=0):
    result2 = firebase.get('/updatecheck/data/',None)
    if result2!=None :
        result2 = firebase.delete('/updatecheck/data/',None)     
    firebase.post('/updatecheck/data', {'check':"Action information does not match.Please press update button."})
elif(picture!=0):
    result2 = firebase.get('/updatecheck/data/',None)
    if result2!=None :
        result2 = firebase.delete('/updatecheck/data/',None)     
    firebase.post('/updatecheck/data', {'check':"Object information does not match.Please press update button."})
else:
    result2 = firebase.get('/updatecheck/data/',None)
    if result2!=None :
        result2 = firebase.delete('/updatecheck/data/',None)  
    firebase.post('/updatecheck/data', {'check':"information match"})