# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 19:05:44 2018

@author: MOJI
"""



from firebase import firebase
firebase = firebase.FirebaseApplication('https://dogwood-terra-184417.firebaseio.com/', None)
import sqlite3
conn = sqlite3.connect('C:\Users\MOJI\Desktop\project\Firebase\python\Test_PJ2.db')
cursor = conn.execute("SELECT ID, Name,Day,Month,Year,Hour,Minute from ActionName")
result = firebase.get('/action/data/',None)
numbers = []
N=[]
nm=[]
for row in cursor:
    for i in result :
        name=firebase.get('/action/data/'+i,'action_name')
        if(name==row[1]): #ถ้าชื่อของเว็บตรงกับหุ่น
            #print row[1]
            numbers.append(row[0]) #เก็บค่าไอดีที่ซ้ำ
            N.append(row[1])
        else:
            nm.append(name)
    idAdd = int(row[0]) 
cursor.close()
#print numbers
#print idAdd
lenNum=len(numbers)-1
#print lenNum
#print nm
namecheck = set(nm).difference(N)
namecheck = list(namecheck)
#print namecheck[0]   
cursor1 = conn.execute("SELECT ID, Name,Day,Month,Year,Hour,Minute from ActionName")    
for n in numbers:
    #print n
    #print numbers[lenNum]  
    for row in cursor1:
        #print row[0]
        if(row[0]!=n and row[0]!=numbers[lenNum]): #หาไอดีไม่ซ้ำจากฝั่งหุ่นยนต์
            #print row[0]
            cursor2 = conn.execute("SELECT ID, stepAction,M1,M2,M3,M4,M5,M6,M7,M8 from Action_Robot")
            for row2 in cursor2:
                if(row[0]==row2[0]):
                    firebase.post('/action/data', {'action_name': row[1],'step':row2[1],'motor': "[ "+str(row2[2])+","+str(row2[3])+","+str(row2[4])+","+str(row2[5])+","+str(row2[6])+","+str(row2[7])+","+str(row2[8])+","+str(row2[9])+" ]",'date':str(row[2])+","+str(row[3])+","+str(row[4])+","+str(row[5])+","+str(row[6])})
            cursor2.close()
        else : #หาตัวซ้ำเพื่อเช็คที่หน้าเว็บ
            #print numbers[lenNum]
            lenNm=len(namecheck)
            #print lenNm  
            if(lenNm!=0):
                nc = str(namecheck[0])
                #print nc
                del namecheck[0]
            else:
                nc = None
            
            ch = 0
            for i in result :
                resultchek = firebase.get('/action/data/'+i,None)
                #print resultchek
                if(resultchek!=None):
                    name=firebase.get('/action/data/'+i,'action_name')
                    motor=firebase.get('/action/data/'+i,'motor')
                    motor = str(motor)
                    motor = motor.split() #ตัดช่องว่าง
                    listmotor = motor[1].split(",")
                    date=firebase.get('/action/data/'+i,'date')
                    listdate = date.split(",")
                    #print(type(listdate[0]))
                    step =firebase.get('/action/data/'+i,'step')
                    if(str(row[1])!=str(name) and nc == name): #ถ้าตัวซ้ำไม่ตรงกับหน้าเว็บ
                        #print 'aaa',name
                        #print 'bbb',row[1]
                        #print 'aa',name
                        #print step
                        if(step=='1'):
                            idAdd=idAdd+1
                            #print 'id',idAdd
                            #print name,listdate,idAdd
                            #with sqlite3.connect('C:\Users\MOJI\Desktop\project\Firebase\python\Test_PJ2.db') as con:
                                #cur3 = con.cursor1()
                                #con.execute('insert into ActionName values (?,?,?,?,?,?,?)',(3,'aaa',1,2,3,4,5))
                                #cur3.execute('insert into ActionName values (?,?,?,?,?,?,?)',(idAdd,str(name),int(listdate[0]),int(listdate[1]),int(listdate[2]),int(listdate[3]),int(listdate[4])))
                                #cur3.execute('insert into Action_Robot values (?,?,?,?,?,?,?,?,?,?)',(idAdd,int(step),int(listmotor[0]),int(listmotor[1]),int(listmotor[2]),int(listmotor[3]),int(listmotor[4]),int(listmotor[5]),int(listmotor[6]),int(listmotor[7])))
                                
                        else:
                            if(ch == 1):
                                idadd = idAdd-1
                            else:
                                idadd = idAdd
                            print name,listdate,idadd,step
                    elif(str(row[1])==str(name)) :
                        print 'aaaaaaaaaaaaaaaaaa'
                        print name #เว็บ
                        print row[1] #หุ่น
                        s=0
                        f=0
                        yearF=int(listdate[2])
                        yearS=int(row[4])
                        monthF=int(listdate[1])
                        monthS=int(row[3])
                        dayF=int(listdate[0])
                        dayS=int(row[2])
                        hourF=int(listdate[3])
                        hourS=int(row[5])
                        minuteF=int(listdate[4])
                        minuteS=int(row[6])
                        if(yearF == yearS): #ปีเท่ากัน
                            if(monthF == monthS):#เดือนเท่ากัน
                                if(dayF == dayS):#วันเท่ากัน
                                    if(hourF == hourS):#ชั่วโมงเท่ากัน
                                        if(minuteF==minuteS):
                                            print 'equal'
                                        elif(minuteF > minuteS):
                                            print "dayF"
                                            f = step
                                        else :
                                            s = step
                                            print "minuteS"
                                            print (minuteS)   
                                    elif(hourF > hourS):
                                        
                                        print "hourF"
                                        f = step
                                    else:
                                        s = step
                                        print (hourS) 
                                        print "hourS"
                                elif(dayF > dayS):
                                    print "dayF"
                                    f = step
                                else:
                                    s = step
                                    print "dayS"
    
                            elif(monthF > monthS):
                                print "monthF"
                                f = step
                            else: 
                                s = step
                                #print (monthS)
                        elif(yearF > yearS):
                            print "yearF"
                            #print (yearF)
                            f = step
                        else: 
                            s = step
                            print "yearS"
                            #print (yearS)
    
                        if(step==s):
                            for j in result :
                                result3 = firebase.get('/action/data/'+j,'action_name')
                                if(str(result3)==str(row[1])):
                                    result4 = firebase.delete('/action/data/'+j,None)
                                    #numbers.remove(row[0]);
                            cursor3 = conn.execute("SELECT ID, stepAction,M1,M2,M3,M4,M5,M6,M7,M8 from Action_Robot")
                            for row3 in cursor3:
                                if(row2[0]==row3[0]):
                                    firebase.post('/action/data', {'action_name': row[1],'step':row3[1],'motor': "[ "+str(row3[2])+","+str(row3[3])+","+str(row3[4])+","+str(row3[5])+","+str(row3[6])+","+str(row3[7])+","+str(row3[8])+","+str(row3[9])+" ]",'date':str(row[2])+","+str(row[3])+","+str(row[4])+","+str(row[5])+","+str(row[6])})
                        elif(step==f):
                            print 'id',idAdd
                            print name
                            if(step=='1'):
                                idAdd=idAdd+1
                                #print 'id',idAdd
                                print name,listdate,idAdd
                            else:
                                print name,listdate,idAdd,step
                            ch=1
                            

print "Operation done successfully";
conn.close()
            
#############picture###########    

from firebase import firebase
from google.cloud import storage
client = storage.Client.from_service_account_json('C:\Users\MOJI\Desktop\project\dogwood-terra-184417-firebase-adminsdk-vy9o9-765ac92f9f.json')
bucket = client.get_bucket('dogwood-terra-184417.appspot.com')
firebase = firebase.FirebaseApplication('https://dogwood-terra-184417.firebaseio.com/', None)
import sqlite3
conn = sqlite3.connect('C:\Users\MOJI\Desktop\project\Firebase\python\Test_PJ2.db')
cursor = conn.execute("SELECT ID,Name,day,month,year,hour,minute from call_Detect")
result = firebase.get('/object/data/',None)
n = []
Na=[]
for row in cursor:
    for i in result :
        name=firebase.get('/object/data/'+i,'object_Name')
        if(name==row[1]):
            print 'a'
        else:
            n.append(name)
            Na.append(row[1])
#print n
#print Na 
cursor.close()
nobcheck = set(n).difference(Na)
nobcheck = list(nobcheck)
ln=len(nobcheck)
#print nobcheck
nobcheck2 = set(Na).difference(n)
nobcheck2 = list(nobcheck2)
ln2=len(nobcheck2)
#print nobcheck2
count=0
count2=0
for i in result :
        name=firebase.get('/object/data/'+i,'object_Name')
        if(count<ln):
            if(name==nobcheck[count]):
                print name   
                count+=1 
                with sqlite3.connect('C:\Users\MOJI\Desktop\project\Firebase\python\Test_PJ2.db') as con:
                                #cur3 = con.cursor1()
                                con.execute('insert into ActionName values (?,?,?,?,?,?,?)',(count,name,1,2,3,4,5))
                
cursor = conn.execute("SELECT ID, Name,K,day,month,year,hour,minute from call_Detect")
for row in cursor:
    if(count2<ln2):
        if(row[1]==nobcheck2[count2]):
            print row[1]
            day=row[3]
            month=row[4]
            year=row[5]
            hour=row[6]
            minute=row[7]
            print day,month,year,hour,minute
            blob = bucket.blob('aaaaa.jpg')
            blob.upload_from_filename(filename ='C:\Users\MOJI\Pictures\8121cc95-36ce-405f-af9b-e67c8589242d.jpg')
            firebase.post('/object/data',{'object_Name':'aaaa','thumbnail':'https://firebasestorage.googleapis.com/v0/b/dogwood-terra-184417.appspot.com/o/'+'8121cc95-36ce-405f-af9b-e67c8589242d.jpg'+'?alt=media','date':str(day)+','+str(month)+','+str(year)+','+str(hour)+','+str(minute)})
            count+=1

###################### motor #################
            
from firebase import firebase
from google.cloud import storage
client = storage.Client.from_service_account_json('C:\Users\MOJI\Desktop\project\dogwood-terra-184417-firebase-adminsdk-vy9o9-765ac92f9f.json')
bucket = client.get_bucket('dogwood-terra-184417.appspot.com')
firebase = firebase.FirebaseApplication('https://dogwood-terra-184417.firebaseio.com/', None)
import sqlite3
conn = sqlite3.connect('C:\Users\MOJI\Desktop\project\Firebase\python\Test_PJ2.db')
cursor = conn.execute("SELECT ID,Name,day,month,year,hour,minute from ActionName")
result = firebase.get('/action/data/',None)
nM = []
NaM=[]
sameD=[]
sameM=[]
sameY=[]
sameH=[]
sameMi=[]
id1=[]
maxid=0
for row in cursor:
    id1.append(row[0])
    NaM.append(row[1])
    sameD.append(row[2])
    sameM.append(row[3])
    sameY.append(row[4])
    sameH.append(row[5])
    sameMi.append(row[6])
    if(row[0]>maxid):
        maxid=row[0]
cursor.close()
print NaM
print sameY
for i in result :
    name=firebase.get('/action/data/'+i,'action_name')
    nM.append(name)
#print sameNM,sameDt
#print lnsameNM
#print n
#print Na 
#print maxid
maxid = maxid+1
cursor.close()
nMocheck = set(nM).difference(NaM)
nMocheck = list(nMocheck)
ln3=len(nMocheck)
#print ln3
#print nMocheck
nMocheck2 = set(NaM).difference(nM)
nMocheck2 = list(nMocheck2)
ln4=len(nMocheck2)
#print ln4,nMocheck2
ln5=len(NaM)
count3=0
count4=0
count5=0

for j in range(ln5):
    for i in result :
        resultchek = firebase.get('/action/data/'+i,None)
        if(resultchek!=None):
            name=firebase.get('/action/data/'+i,'action_name')
            motor=firebase.get('/action/data/'+i,'motor')
            motor = str(motor)
            motor = motor.split() #ตัดช่องว่าง
            listmotor = motor[1].split(",")
            date=firebase.get('/action/data/'+i,'date')
            listdate = date.split(",")
            #print(type(listdate[0]))
            step =firebase.get('/action/data/'+i,'step')
            yearF=int(listdate[2])
            if(count5<ln5):
                if(name==NaM[count5]):
                    print "..................."
                    print name
                    print sameY[count5]
                    print NaM[count5]
                    print yearF
                    s=0
                    f=0
                    yearF=int(listdate[2])
                    monthF=int(listdate[1])
                    dayF=int(listdate[0])
                    hourF=int(listdate[3])
                    minuteF=int(listdate[4])
                    if(yearF == sameY[count5]): #ปีเท่ากัน
                        if(monthF == sameM[count5]):#เดือนเท่ากัน
                            if(dayF == sameD[count5]):#วันเท่ากัน
                                if(hourF == sameH[count5]):#ชั่วโมงเท่ากัน
                                    if(minuteF==sameMi[count5]):
                                        print 'equal'
                                    elif(minuteF > sameMi[count5]):
                                        print "dayF"
                                        f = step
                                    else :
                                        s = step
                                        print "minuteS"
                                        print (minuteS)   
                                elif(hourF > sameH[count5]):
                                    print "hourF"
                                    f = step
                                else:
                                    s = step
                                    print (hourS) 
                                    print "hourS"
                            elif(dayF > sameD[count5]):
                                print "dayF"
                                f = step
                            else:
                                s = step
                                print "dayS"
                        elif(monthF > sameM[count5]):
                            print "monthF"
                            f = step
                        else: 
                            s = step
                            #print (monthS)
                    elif(yearF > sameY[count5]):
                        print "yearF"
                        #print (yearF)
                        f = step
                    else: 
                        s = step
                        print "yearS"
                        #print (yearS)
                    if(step==s):
                        for j in result :
                            result3 = firebase.get('/action/data/'+j,'action_name')
                            if(str(result3)==str(NaM[count5])):
                                result4 = firebase.delete('/action/data/'+j,None)
                        cursor3 = conn.execute("SELECT ID, stepAction,M1,M2,M3,M4,M5,M6,M7,M8 from Action_Robot")
                        for row3 in cursor3:
                            if(id1[count5]==row3[0]):
                                firebase.post('/action/data', {'action_name': NaM[count5],'step':row3[1],'motor': "[ "+str(row3[2])+","+str(row3[3])+","+str(row3[4])+","+str(row3[5])+","+str(row3[6])+","+str(row3[7])+","+str(row3[8])+","+str(row3[9])+" ]",'date':str(row[2])+","+str(row[3])+","+str(row[4])+","+str(row[5])+","+str(row[6])})
                        cursor3.close()     
                    elif(step==f):
                        print name,"aaaa",step,'bbb',id1[count5]
                        conn = sqlite3.connect('C:\Users\MOJI\Desktop\project\Firebase\python\Test_PJ2.db')
                        conn.execute("DELETE from ActionName where ID = "+str(id1[count5])+";")
                        conn.commit()
                        conn.close()
                        conn = sqlite3.connect('C:\Users\MOJI\Desktop\project\Firebase\python\Test_PJ2.db')
                        conn.execute("DELETE from Action_Robot where ID = "+str(id1[count5])+";")
                        conn.commit()
                        conn.close()
                        if(str(step)=='1'):
                            with sqlite3.connect('C:\Users\MOJI\Desktop\project\Firebase\python\Test_PJ2.db') as con:
                                cur3 = con.cursor()
                                cur3.execute('insert into ActionName values (?,?,?,?,?,?,?)',(int(id1[count5]),str(name),int(listdate[0]),int(listdate[1]),int(listdate[2]),int(listdate[3]),int(listdate[4])))
                    
                        with sqlite3.connect('C:\Users\MOJI\Desktop\project\Firebase\python\Test_PJ2.db') as con:
                                cur4 = con.cursor()
                                cur4.execute('insert into Action_Robot values (?,?,?,?,?,?,?,?,?,?)',(int(id1[count5]),int(step),int(listmotor[0]),int(listmotor[1]),int(listmotor[2]),int(listmotor[3]),int(listmotor[4]),int(listmotor[5]),int(listmotor[6]),int(listmotor[7])))
    count5+=1

for j in range(ln3):
    for i in result :
            name=firebase.get('/action/data/'+i,'action_name')
            motor=firebase.get('/action/data/'+i,'motor')
            motor = str(motor)
            motor = motor.split() #ตัดช่องว่าง
            listmotor = motor[1].split(",")
            date=firebase.get('/action/data/'+i,'date')
            listdate = date.split(",")
            #print(type(listdate[0]))
            step =firebase.get('/action/data/'+i,'step')
            #print name
            if(count3<ln3):
                if(name==nMocheck[count3]):  
                    print name,listdate,count3,maxid
                    if(str(step)=='1'):
                        with sqlite3.connect('C:\Users\MOJI\Desktop\project\Firebase\python\Test_PJ2.db') as con:
                                cur3 = con.cursor()
                                cur3.execute('insert into ActionName values (?,?,?,?,?,?,?)',(maxid,str(name),int(listdate[0]),int(listdate[1]),int(listdate[2]),int(listdate[3]),int(listdate[4])))
                    
                    with sqlite3.connect('C:\Users\MOJI\Desktop\project\Firebase\python\Test_PJ2.db') as con:
                                cur4 = con.cursor()
                                cur4.execute('insert into Action_Robot values (?,?,?,?,?,?,?,?,?,?)',(maxid,int(step),int(listmotor[0]),int(listmotor[1]),int(listmotor[2]),int(listmotor[3]),int(listmotor[4]),int(listmotor[5]),int(listmotor[6]),int(listmotor[7])))    
    count3+=1
    maxid+1
  
cursor = conn.execute("SELECT ID, Name,Day,Month,Year,Hour,Minute from ActionName")   
for j in range(ln4):
    for row in cursor:
        if(count4<ln4):
            if(row[1]==nMocheck2[count4]):
                print row[1]
                day=row[2]
                month=row[3]
                year=row[4]
                hour=row[5]
                minute=row[6]
                print day,month,year,hour,minute
                cursor2 = conn.execute("SELECT ID, stepAction,M1,M2,M3,M4,M5,M6,M7,M8 from Action_Robot")
                for row2 in cursor2:
                    if(row[0]==row2[0]):
                        firebase.post('/action/data', {'action_name': row[1],'step':row2[1],'motor': "[ "+str(row2[2])+","+str(row2[3])+","+str(row2[4])+","+str(row2[5])+","+str(row2[6])+","+str(row2[7])+","+str(row2[8])+","+str(row2[9])+" ]",'date':str(row[2])+","+str(row[3])+","+str(row[4])+","+str(row[5])+","+str(row[6])})
                cursor2.close()
    count4+=1
