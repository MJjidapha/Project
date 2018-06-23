import sqlite3
from firebase import firebase
firebase = firebase.FirebaseApplication('https://dogwood-terra-184417.firebaseio.com/', None)
def search_object_Train():
    with sqlite3.connect("db1.db") as con:
        cur = con.cursor()
        #cur.execute('SELECT * FROM object_Train WHERE name=?',(name,))
        try :
            cur.execute("SELECT *FROM action")
            rows = cur.fetchall()
           
            for element in rows:
                list = element
                print list
                firebase.post('/action/data', {'action_name': str(list[0]),'motor':str(list[2:10]),'step':list[1]})
        except :
            return "None"
       # return cur.fetchone() # None OR (u'Ball', 2)
       
search_object_Train()

#list = search_object_Train("grab")
#print list[2:10]



from firebase import firebase
firebase = firebase.FirebaseApplication('https://dogwood-terra-184417.firebaseio.com/', None)
result = firebase.get('/Gauges/data/',None)
if result!="None" :
    result = firebase.delete('/Gauges/data/',None)     
firebase.post('/Gauges/data', {'Battery':float(65.22),'temperature':float(20.55)})