def storageTest(name,num):
    from firebase import firebase
    from google.cloud import storage
    name = str(name)
    num = str(num)
    firebase = firebase.FirebaseApplication('https://dogwood-terra-184417.firebaseio.com/', None)
    client = storage.Client.from_service_account_json(
        '/home/kns/PycharmProjects/Aj/AJ2/dogwood-terra-184417-firebase-adminsdk-vy9o9-765ac92f9f.json')
    bucket = client.get_bucket('dogwood-terra-184417.appspot.com')
    objectFirebese = str(name)
    pathPicCom = "/home/kns/PycharmProjects/Aj/AJ2/pic/" + objectFirebese + "/" + objectFirebese + num +"_1.jpg"
    print(pathPicCom)
    blob = bucket.blob(objectFirebese+ num + ".jpg")
    blob.upload_from_filename(filename=pathPicCom)
    firebase.post('/object/data', {'object_Name': objectFirebese+str(num),
                                   'thumbnail': 'https://firebasestorage.googleapis.com/v0/b/dogwood-terra-184417.appspot.com/o/' + objectFirebese +num+ '.jpg' + '?alt=media'})
