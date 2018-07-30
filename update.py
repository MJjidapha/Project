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
#print numbers
cursor1 = conn.execute("SELECT ID, Name,Day,Month,Year,Hour,Minute from ActionName")
#print idAdd
lenNum=len(numbers)-1
#print lenNum
#print nm
namecheck = set(nm).difference(N)
namecheck = list(namecheck)
#print namecheck[0]      
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
                            print name,listdate,idAdd
                            with sqlite3.connect('C:\Users\MOJI\Desktop\project\Firebase\python\Test_PJ2.db') as con:
                                cur3 = con.cursor()
                                cur3.execute('insert into ActionName values (?,?,?,?,?,?,?)',(3,'aaa',1,2,3,4,5))
                                #cur3.execute('insert into ActionName values (?,?,?,?,?,?,?)',(idAdd,str(name),int(listdate[0]),int(listdate[1]),int(listdate[2]),int(listdate[3]),int(listdate[4])))
                                #cur3.execute('insert into Action_Robot values (?,?,?,?,?,?,?,?,?,?)',(idAdd,int(step),int(listmotor[0]),int(listmotor[1]),int(listmotor[2]),int(listmotor[3]),int(listmotor[4]),int(listmotor[5]),int(listmotor[6]),int(listmotor[7])))
                                
                        else:
                            if(ch == 1):
                                idadd = idAdd-1
                            else:
                                idadd = idAdd
                            print name,listdate,idadd,step
                    elif(str(row[1])==str(name)) :
                        #print listdate[1] #เว็บ
                        #print row[3] #หุ่น
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
                        result2 = firebase.get('/action/data/'+i,'action_name')
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
firebase = firebase.FirebaseApplication('https://dogwood-terra-184417.firebaseio.com/', None)
import sqlite3
conn = sqlite3.connect('C:\Users\MOJI\Desktop\project\Firebase\python\Test_PJ2.db')
cursor = conn.execute("SELECT ID, Name from call_Detect")
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
        
cursor = conn.execute("SELECT ID, Name from call_Detect")
for row in cursor:
    if(count2<ln2):
        if(row[1]==nobcheck2[count2]):
            print row[1]
            count+=1