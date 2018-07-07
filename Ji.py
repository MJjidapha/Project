############################################# IMPORT Tensorflow #########################################################

PATH = "/home/kns/PycharmProjects/Aj/AJ2"
Video = 1
import time
from espeak import espeak

espeak.set_parameter(espeak.Parameter.Pitch, 60)
espeak.set_parameter(espeak.Parameter.Rate, 110)
espeak.set_parameter(espeak.Parameter.Range, 600)
espeak.synth("Hey Guys My name is Jerry")
time.sleep(2)

import numpy as np
import os
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# -----------------Model-----------------


import imutils
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler

############################################  IMPORT Sphinx  ########################################################


from os import path
import pyaudio
from pocketsphinx.pocketsphinx import *
from sphinxbase.sphinxbase import *
from check4 import *
from manageDB import *

MODELDIR = PATH + "/model_LG"
DATADIR = PATH + "/dataLG"

config = Decoder.default_config()
config.set_string('-logfn', '/dev/null')
config.set_string('-hmm', path.join(MODELDIR, 'en-us/en-us'))
config.set_string('-lm', path.join(MODELDIR, 'en-us/en-us.lm.bin'))
config.set_string('-dict', path.join(MODELDIR, 'en-us/cmudict-en-us.dict'))
decoder = Decoder(config)

# Switch to JSGF grammar
jsgf = Jsgf(path.join(DATADIR, 'sentence.gram'))
rule = jsgf.get_rule('sentence.move')  # >> public <move>
fsg = jsgf.build_fsg(rule, decoder.get_logmath(), 7.5)
fsg.writefile('sentence.fsg')

decoder.set_fsg("sentence", fsg)
decoder.set_search("sentence")

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
stream.start_stream()

in_speech_bf = False
decoder.start_utt()

STPindex = 0
STPname = ""


#################################### CAP IMAGE ###############################################


def capture(namePath, obj_name, count):
    print "CAPPP"
    import time
    start = time.time()
    time.clock()
    elapsed = 0
    i = 0
    seconds = 100  # 20 S.
    cap = cv2.VideoCapture(Video)
    # Running the tensorflow session
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            ret = True
            while (ret):
                ret, image_np = cap.read()
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                elapsed = int(time.time() - start)
                # print "EP : ", elapsed

                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array2(image_np,
                                                                    np.squeeze(boxes),
                                                                    np.squeeze(classes).astype(np.int32),
                                                                    np.squeeze(scores),
                                                                    category_index,
                                                                    use_normalized_coordinates=True,
                                                                    line_thickness=8)

                cv2.imshow('image', cv2.resize(image_np, (640, 480)))

                if (elapsed % 10 == 0):
                    i += 1

                    try:
                        y = int(vis_util.f.getYmin() * 479.000)
                        yh = int(vis_util.f.getYmax() * 479.000)
                        x = int(vis_util.f.getXmin() * 639.000)
                        xh = int(vis_util.f.getXmax() * 639.000)
                        print y, " ", yh, " ", x, " ", xh
                        cv2.imshow('RGB image', image_np)

                        params = list()
                        # 143 : 869 // 354 :588
                        # 120:420, 213:456
                        crop_img = image_np[y:yh, x:xh]

                        cv2.imwrite(namePath + obj_name + str(count) + "_" + str(i) + ".jpg",
                                    crop_img, params)
                        # i+=1print "Do you want to save?" + "look : " + STPname
                        espeak.synth("continue")
                        time.sleep(2)
                        # print "OK cap"
                        cv2.destroyAllWindows()

                    except:
                        print "CAP Finished!"
                        print "no image PASS"

                if (elapsed >= seconds):
                    cv2.destroyAllWindows()
                    break

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
    cap.release()
    cv2.destroyAllWindows()


########################################################################################################
########################################## SAVE MODEL ##################################################

def save_model():
    R = 20
    train_path = PATH + "/pic"
    training_names = os.listdir(train_path)
    image_paths = []
    image_classes = []  ## 00000,111111,2222,33333
    class_id = 0
    for training_name in training_names:
        dir = os.path.join(train_path, training_name)
        class_path = imutils.imlist(dir)

        image_paths += class_path
        image_classes += [class_id] * len(class_path)
        class_id += 1

    # Create feature extraction and keypoint detector objects
    # print image_classes," imP :",image_paths
    sift = cv2.xfeatures2d.SIFT_create()

    # List where all the descriptors are stored
    des_list = []
    for x in range(0, R):
        # print x
        for image_path in image_paths:
            # print image_path
            try:
                im = cv2.imread(image_path)
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                kpts, des = sift.detectAndCompute(gray, None)
                des_list.append((image_path, des))
            except:
                pass

    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[1:]:
        try:
            descriptors = np.vstack((descriptors, descriptor))
        except:
            pass

    # Perform k-means clustering
    k = 100
    vec, variance = kmeans(descriptors, k, 1)
    # print len(voc[4]) #128 len(voc)=100

    # Calculate the histogram of features
    im_features = np.zeros((len(image_paths), k), "float32")  # len(ALL pic) >> [0000000][00000]...

    for i in xrange(len(image_paths)):
        try:
            words, distance = vq(des_list[i][1], vec)
            for w in words:
                im_features[i][w] += 1

        # Scaling the words
        except:
            print "235"
            pass
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)

    # Train the Linear SVM
    clf = LinearSVC()
    clf.fit(im_features, np.array(image_classes))
    # Save the SVM

    joblib.dump((clf, training_names, stdSlr, k, vec), "train.pkl", compress=3)
    updateSAVE_Train()
    print "SAVE MODEL"


###########################################################################################################
import sys
import time
import rospy
import sqlite3
from std_msgs.msg import UInt8
from std_msgs.msg import UInt16MultiArray
from std_msgs.msg import String


def callback_Talk(msg):  # insert ActionName
    with sqlite3.connect("Test_PJ2.db") as con:
        cur = con.cursor()
        try:
            cur.execute('SELECT * FROM ActionName')
            rows = cur.fetchall()
            lenR = len(rows)
            cur.execute('insert into ActionName (Name,ID) values (?,?)', (msg, lenR + 1,))
        except:
            return "Name Error"


def selectID_AcName(Action):
    with sqlite3.connect("Test_PJ2.db") as con:
        cur = con.cursor()
        try:
            cur.execute("Select ID from ActionName where Name = ?", (Action,))
            row = cur.fetchone()
            for element in row:
                id = element
                return id
        except:
            return "Error"


def select_Buffer():
    list = []
    with sqlite3.connect("Test_PJ2.db") as con:
        cur = con.cursor()
        try:
            cur.execute("Select ID,M1,M2,M3,M4,M5,M6,M7,M8 from Buffer_Action")
            row = cur.fetchall()
            for element in row:
                motor = element
                list.append(motor)
            return list
        except:
            return "Null"


def del_buff():
    with sqlite3.connect("Test_PJ2.db") as con:
        sql_cmd = """
        delete from Buffer_Action
        """
        con.execute(sql_cmd)


########### move arm ###############

def sendCmd2(msg):
    # pub1 = rospy.Publisher("state", UInt8, queue_size=1, tcp_nodelay=False)
    pub2 = rospy.Publisher("command", UInt8, queue_size=1, tcp_nodelay=False)

    rospy.init_node('Move', anonymous=True)
    # pub1.publish(0)
    pub2.publish(int(msg))


def smartDelay(x, y):
    tmp = []
    maxDiff = 0
    for i in range(0, len(x)):
        diff = abs(x[i] - y[i])
        tmp.append(abs(x[i] - y[i]))
        if (diff > maxDiff):
            maxDiff = diff
    print(x)
    print(y)
    print(tmp)
    print(max(tmp))
    print(maxDiff)
    print(max(tmp) / 16.)

    time.sleep(max(tmp) / 16.);


def send(msg):
    pub1 = rospy.Publisher("state", UInt8, queue_size=1, tcp_nodelay=False)
    pub2 = rospy.Publisher("movement", UInt16MultiArray, queue_size=5, tcp_nodelay=False)

    rospy.init_node('Move', anonymous=True)

    pub1.publish(0)
    time.sleep(0.1)
    pub2.publish(msg)


# rospy.wait_for_message("state", UInt8)  ## wait for the acknowledgment, with timeout


def talker2(msg2):
    pub3 = rospy.Publisher('walker', String, queue_size=10)

    rospy.init_node('Move', anonymous=True)

    pub3.publish(String(msg2))


################################################## Keep #####################################################

def keep_First_Home():
    # home
    with sqlite3.connect("Test_PJ2.db") as con:
        cur = con.cursor()
        cur.execute('select M1,M2,M3,M4,M5,M6,M7,M8 from Action_Robot where ID = 1')
        row1 = cur.fetchall()
        for element1 in row1:
            print(element1)
            msg = UInt16MultiArray(data=element1)
            send(msg)
    # sendCmd1(1)


def insert_name():
    try:
        with sqlite3.connect("Test_PJ2.db") as con:
            cur = con.cursor()
            lenObj = int(lenDB("Test_PJ2.db", "SELECT * FROM ActionName"))  # count ROWs
            cur.execute('insert into ActionName (ID,Name) values (?,?)',
                        (lenObj + 1, STPname))
            print(STPname)
    except:
        print "Action in table!!!"
    print("SAVE NAME TO Table Main_action")


################################################### Detect ##########################################
import time

start = time.time()
time.clock()
elapsed = 0
seconds = 20  # 20 S.

import numpy as np
import os

import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# intializing the web camera device

import cv2


def detectBOW4(name):
    start = time.time()
    time.clock()
    elapsed = 0
    seconds = 10  # 20 S.
    i = 0
    K = 0
    Ymax = 0
    c = 1
    cap = cv2.VideoCapture(Video)
    vis_util.f.setPredic("")
    objName = "None"
    with sqlite3.connect("Test_PJ2.db") as con:
        cur = con.cursor()
        try:
            cur.execute("UPDATE call_Detect SET Name=?,K=? ,Ymax =? WHERE ID = 1", (objName, 0, 0))
        except:
            print "Line 343 "
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            ret = True
            while (ret):
                ret, image_np = cap.read()
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                vis_util.visualize_boxes_and_labels_on_image_array(image_np,
                                                                   np.squeeze(boxes),
                                                                   np.squeeze(classes).astype(np.int32),
                                                                   np.squeeze(scores),
                                                                   category_index,
                                                                   use_normalized_coordinates=True,
                                                                   line_thickness=8)
                elapsed = int(time.time() - start)
                cv2.imshow('image', cv2.resize(image_np, (640, 480)))
                st = vis_util.f.getPredic()
                if st != "":
                    st = st.split("#")
                    objName = st[0]
                    c += 1
                    if objName == str(name):
                        i += 1
                        st2 = st[1].split(",")
                        Xmax = st2[3]
                        Xmin = st2[2]
                        Ymax = st2[1]
                        K = (int(Xmax) + int(Xmin)) / 2
                        st3 = objName + " " + str(K)
                        print st3, ' ', Ymax

                if (elapsed >= seconds):
                    print "i= ", i
                    if (i > 4):
                        with sqlite3.connect("Test_PJ2.db") as con:
                            cur = con.cursor()
                            print "update ", name, " K = ", K, "Ymax = ", Ymax
                            cur.execute("UPDATE call_Detect SET Name=?,K=? , Ymax=? WHERE ID = 1", (name, K, Ymax))
                        # cv2.imshow('image', cv2.resize(image_np, (640, 480)))
                    else:
                        print "I can not see (no update)"
                    break
                    cv2.destroyAllWindows()  # if (elapsed >= seconds):

                #  return st3
                #  cv2.destroyAllWindows()

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
    cap.release()
    cv2.destroyAllWindows()


##################################################################################################################
##################################################################################################################
############################################## MAIN MOTOR CONTROL ################################################
##################################################################################################################
##################################################################################################################

def Contorl_Basic_FGB(v):
    # Forword >> Grab >> Backword
    j = 1
    talker2("startDefult")
    time.sleep(6)
    with sqlite3.connect("Test_PJ2.db") as con:
        cur = con.cursor()
        cur.execute(
            'SELECT * from Action_Robot INNER JOIN ActionName ON Action_Robot.ID = ActionName.ID WHERE Name = ?', (v,))
        row = cur.fetchall()

        for element in row:
            if (j == 1):
                with sqlite3.connect("Test_PJ2.db") as con:
                    cur11 = con.cursor()
                    cur11.execute(
                        'SELECT Action_Robot.M1,Action_Robot.M2,Action_Robot.M3,Action_Robot.M4,Action_Robot.M5,Action_Robot.M6,Action_Robot.M7,Action_Robot.M8 FROM Action_Robot INNER JOIN ActionName ON Action_Robot.ID = ActionName.ID WHERE Name = ? AND stepAction = ?',
                        (v, j))
                    row11 = cur11.fetchall()
                    for element11 in row11:
                        print(element11)
                    msg1 = UInt16MultiArray(data=element11)
                    send(msg1)
                    j = j + 1

            elif (j != 1):
                with sqlite3.connect("Test_PJ2.db") as con:
                    cur1 = con.cursor()
                    cur1.execute(
                        'SELECT Action_Robot.M1,Action_Robot.M2,Action_Robot.M3,Action_Robot.M4,Action_Robot.M5,Action_Robot.M6,Action_Robot.M7,Action_Robot.M8 FROM Action_Robot INNER JOIN ActionName ON Action_Robot.ID = ActionName.ID WHERE Name = ? AND stepAction = ?',
                        (v, j - 1))
                    row1 = cur1.fetchall()
                    for element1 in row1:
                        x = element1

                    cur2 = con.cursor()
                    cur2.execute(
                        'SELECT Action_Robot.M1,Action_Robot.M2,Action_Robot.M3,Action_Robot.M4,Action_Robot.M5,Action_Robot.M6,Action_Robot.M7,Action_Robot.M8 FROM Action_Robot INNER JOIN ActionName ON Action_Robot.ID = ActionName.ID WHERE Name = ? AND stepAction = ?',
                        (v, j))
                    row2 = cur2.fetchall()
                    for element2 in row2:
                        y = element2

                smartDelay(x, y)
                msg2 = UInt16MultiArray(data=y)
                send(msg2)
                j = j + 1
    time.sleep(1)
    talker2("Backward")
    time.sleep(5)


def Control_Basic_Move(v, center1):
    j = 1
    center1 = str(center1)
    print center1, "Move ROBOT ", type(center1)
    talker2(center1)
    time.sleep(20)

    with sqlite3.connect("Test_PJ2.db") as con:
        cur = con.cursor()
        cur.execute(
            'SELECT * from Action_Robot INNER JOIN ActionName ON Action_Robot.ID = ActionName.ID WHERE Name = ?', (v,))
        row = cur.fetchall()

        for element in row:
            if (j == 1):
                with sqlite3.connect("Test_PJ2.db") as con:
                    cur11 = con.cursor()
                    cur11.execute(
                        'SELECT Action_Robot.M1,Action_Robot.M2,Action_Robot.M3,Action_Robot.M4,Action_Robot.M5,Action_Robot.M6,Action_Robot.M7,Action_Robot.M8 FROM Action_Robot INNER JOIN ActionName ON Action_Robot.ID = ActionName.ID WHERE Name = ? AND stepAction = ?',
                        (v, j))
                    row11 = cur11.fetchall()
                    for element11 in row11:
                        print(element11)
                    msg1 = UInt16MultiArray(data=element11)
                    send(msg1)
                    time.sleep(2)

                    j = j + 1

            elif (j != 1):
                with sqlite3.connect("Test_PJ2.db") as con:
                    cur1 = con.cursor()
                    cur1.execute(
                        'SELECT Action_Robot.M1,Action_Robot.M2,Action_Robot.M3,Action_Robot.M4,Action_Robot.M5,Action_Robot.M6,Action_Robot.M7,Action_Robot.M8 FROM Action_Robot INNER JOIN ActionName ON Action_Robot.ID = ActionName.ID WHERE Name = ? AND stepAction = ?',
                        (v, j - 1))
                    row1 = cur1.fetchall()
                    for element1 in row1:
                        x = element1

                    cur2 = con.cursor()
                    cur2.execute(
                        'SELECT Action_Robot.M1,Action_Robot.M2,Action_Robot.M3,Action_Robot.M4,Action_Robot.M5,Action_Robot.M6,Action_Robot.M7,Action_Robot.M8 FROM Action_Robot INNER JOIN ActionName ON Action_Robot.ID = ActionName.ID WHERE Name = ? AND stepAction = ?',
                        (v, j))
                    row2 = cur2.fetchall()
                    for element2 in row2:
                        y = element2

                smartDelay(x, y)
                msg2 = UInt16MultiArray(data=y)
                send(msg2)
                time.sleep(2)
                j = j + 1

    talker2("Backward")
    time.sleep(5)
    # MOVE ROBOT


def checkLeft():
    talker2("Deflut")
    time.sleep(2)
    talker2("turnLeft45")


def checkRight():
    talker2("Deflut")
    time.sleep(2)
    talker2("turnRight45")


def stat():
    talker2("startDefult")
    time.sleep(6)
    talker2("Backward")


def goToRight():
    talker2("turnRightForword")


def goToLeft():
    talker2("turnLeftForword")


##################################################################################################################
##################################################################################################################

JOB = True
JOB_HowTo_Open = False
STPindex = 0
JERRY = True
TRAIN_DATA_SET = True
check_Go = False
JOB_SAVE = False
TRAIN = False
# talker2("-1")
talker2("Deflut")

#############################################################################################################
#################################### GOOGLE CLOUD STORAGE ##################################################

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




##############################################################################################################
##############################################################################################################



from firebase import firebase
firebase = firebase.FirebaseApplication('https://dogwood-terra-184417.firebaseio.com/', None)

while (True):
    result = firebase.get('/STT/',None)
    result2 = str(result)
    if (result2!="None"):
        result = list(result)
        #print result
        getSTT = firebase.get('/STT/' + result[0], None)
        strDecode = str(getSTT).lower()
        delSTT = firebase.delete('/STT/' + result[0], None)
        print(strDecode)

        if strDecode != '':
                # >>>>>>>>>>>>>>> END <<<<<<<<<<<<<<<<<<<<<<<<<
                if strDecode == 'merry one':
                    print "..."
                # talker2("-1")

                if JOB == True and strDecode[:9] == "this is a":
                    JOB = False
                    TRAIN = True
                    print "\n--------------------this is a----------------------"

                    ST = strDecode[9:]
                    print "Do you want to train", ST, "image (yes or no)"
                    ST = "Do you want to train" + ST + " image"
                    firebase.post('/robotsend/data',{'textrobot':str(ST)})
                    espeak.synth(ST)
                    time.sleep(4)
                    obj_name = get_object_train_Ji(strDecode)  # sentence to word

                elif TRAIN == True and strDecode == "yes":
                    espeak.synth("ok,Please rotate the object. When i say continue")
                    time.sleep(7)
                    talker2("startDefult")
                    time.sleep(6)
                    talker2("Backward")
                    time.sleep(5)
                    print "Speech : ", obj_name
                    # create folder
                    dataset_Path = r'/home/kns/PycharmProjects/Aj/AJ2/pic/' + obj_name
                    p = PATH + "/pic/" + obj_name + "/"

                    if not os.path.exists(dataset_Path):
                        print "New Data"
                        os.makedirs(dataset_Path)
                        capture(p, obj_name, 1)  # capture image for train >> SAVE IMAGE
                        lenObj = int(lenDB("Corpus_Main.db", "SELECT * FROM obj_ALL2"))  # count ROWs
                        insert_object_Train2(obj_name, int(lenObj + 1))  # check Found objects?
                    else:
                        count = int(search_count_Train2(obj_name))
                        capture(p, obj_name, count + 1)  ####cap2
                        update_object_Train2(count + 1, obj_name)  # UPDATE COUNT++

                    JOB = True
                    TRAIN = False
                    pathToFrieBase = p+"1_1"

                    #SEARCH COUNT AND UPDATE TO STORAGE
                    count = int(search_count_Train2(obj_name))
                    storageTest(obj_name,count)

                    print "OK "
                    espeak.synth("OK")
                    time.sleep(2)

                    print "\n------------------------------------------"

                elif TRAIN == True and strDecode == "no":
                    print obj_name
                    JOB = True
                    TRAIN = False
                    print "\n------------------------------------------"



                # >>>>>>> JERRY <<<<<<<<<<<<

                elif JERRY == True and JOB == True and strDecode[:5] == 'jerry':
                    Jsubject = strDecode
                    JOB = False
                    JERRY = False
                    print "\n---------------------jerry---------------------"
                    print '\nStream decoding result:', strDecode
                    obj_name = get_objectJerry(strDecode)
                    ST = strDecode[5:]
                    print "Do you want to", ST, " (yes or no)"
                    ST = "Do you want to" + ST
                    firebase.post('/robotsend/data', {'textrobot': str(ST)})
                    espeak.synth(ST)
                    time.sleep(4)

                elif JERRY == False and strDecode == "yes":
                    print "OK"
                    espeak.synth("OK")
                    time.sleep(1)
                    talker2("startDefult")
                    time.sleep(6)
                    talker2("Backward")
                    time.sleep(5)
                    # keep_First_Home()
                    obj_find = str(search_object_Train(obj_name))  # KNOW
                    print "obj_find"
                    v = get_V(Jsubject)
                    print obj_name, " ", v
                    # sert name
                    check1 = 0

                    with sqlite3.connect("Test_PJ2.db") as con:
                        cur1 = con.cursor()
                        cur1.execute(
                            'Select ID from ActionName where Name = ?', (v,))
                        row1 = cur1.fetchall()
                        for i in row1:
                            check1 = check1 + 1

                    if obj_find != "None" and check1 != 0:
                        detectBOW4(obj_name)
                        time.sleep(2)

                        print obj_name, " and ", v
                        center1 = str(search_callDetect(obj_name))
                        print ">>> call", center1

                        if (center1 != "None"):
                            center1 = int(center1)
                            if center1 > 55 and center1 < 100:
                                Contorl_Basic_FGB(v)
                                print"contorl_Basic"
                            elif center1 <= 55:
                                print "Left"
                                Control_Basic_Move(v, center1)
                            else:
                                Control_Basic_Move(v, center1)
                                print"contorl_Move"

                        if center1 == "None":
                            print "I can not see it..............................1"
                            checkLeft()  # CHECK TO The LEFT @ Roll LEFT
                            time.sleep(2)
                            talker2("Forward2000")
                            time.sleep(2)
                            detectBOW4(obj_name)
                            talker2("Backward2000")
                            time.sleep(2)

                            center1 = str(search_callDetect(obj_name))
                            print obj_name, " and ", v, " ", center1

                            if (center1 != "None"):
                                goToLeft()  # @ TO The LEFT
                                time.sleep(17)
                                detectBOW4(obj_name)  # Detect

                                center1 = str(search_callDetect(obj_name))

                                if (center1 != "None"):
                                    center1 = int(center1)
                                    if center1 > 55 and center1 < 100:
                                        Contorl_Basic_FGB(v)
                                    elif center1 <= 55:
                                        print "left"
                                        Control_Basic_Move(v, center1)

                                    else:
                                        Control_Basic_Move(v, center1)

                            if center1 == "None":
                                print "I can not see it...........................2"  #### if not @ Back <---
                                talker2("turnRight45")  # @ TO Mid , Roll Right
                                time.sleep(2)
                                checkRight()  # CHECK TO The Right @ Roll Right
                                time.sleep(2)

                                j = 1
                                talker2("Forward2000")
                                time.sleep(2)
                                detectBOW4(obj_name)
                                talker2("Backward2000")
                                time.sleep(2)
                                center1 = str(search_callDetect(obj_name))
                                print obj_name, " and ", v, " ", center1

                                if (center1 != "None"):
                                    goToRight()  # @ TO The Right
                                    time.sleep(17)
                                    detectBOW4(obj_name)

                                    center1 = str(search_callDetect(obj_name))
                                    if (center1 != "None"):
                                        center1 = int(center1)
                                        if center1 > 55 and center1 < 100:
                                            print "Basic >>"
                                            Contorl_Basic_FGB(v)
                                        elif center1 <= 55:
                                            print "left"
                                            Control_Basic_Move(v, center1)
                                        else:
                                            print "Move >>"
                                            Control_Basic_Move(v, center1)

                                if center1 == "None":
                                    print "I can not see it"  #### if not @ Back <---
                                    talker2("turnLeft45")  # @ TO Mid , Roll Left
                                    time.sleep(2)

                    print "\n------------------------------------------"
                    JOB = True
                    JERRY = True

                elif JERRY == False and strDecode == "no":
                    keep_First_Home()
                    espeak.synth("ok no")
                    time.sleep(2)
                    print "ok no"
                    print "\n------------------------------------------"
                    JOB = True
                    JERRY = True


                # >>>>>>> ARM <<<<<<<<<<<<
                elif check_Go == False and JOB == True and strDecode[:14] == 'this is how to':
                    JOB = False
                    print "\n------------------------------------------"
                    print '\nStream decoding result:', strDecode
                    STPname = get_V(strDecode)  # grab
                    check = 0

                    with sqlite3.connect("Test_PJ2.db") as con:
                        cur1 = con.cursor()
                        cur1.execute(
                            'Select ID from ActionName where Name = ?', (STPname,))
                        row1 = cur1.fetchall()
                        for i in row1:
                            check = check + 1
                    if (check != 0):
                        check_Go = True

                        keep_First_Home()

                        print "Do you want to save?" + "look : " + STPname
                        st = "Do you want to " + STPname
                        espeak.synth(st)
                        time.sleep(1)

                    else:

                        keep_First_Home()

                        insert_name()

                        print " check=0 , please say next. If you want to teach me"
                        espeak.synth("OK , please say next. If you want to teach me")
                        time.sleep(2)
                        JOB_HowTo_Open = True


                elif check_Go == True and strDecode == "yes let go":

                    sendCmd2(1)
                    print " OK , please say next. If you want to teach me"
                    espeak.synth("OK , please say next. If you want to teach me")
                    time.sleep(2)
                    talker2("startDefult")
                    time.sleep(6)

                    talker2("move")

                    time.sleep(2)
                    JOB_HowTo_Open = True
                    check_Go = False

                elif check_Go == True and strDecode == "no":
                    print "OK stop"
                    espeak.synth("OK stop it")
                    time.sleep(2)

                    check_Go = False
                    JOB_HowTo_Open = False
                    JOB = True


                elif JOB_HowTo_Open == True and strDecode == 'next':
                    talker2("move")
                    JOB = False
                    print 'Stream decoding result:', strDecode
                    STPindex += 1
                    print STPindex, " : ", STPname

                    # send(1)

                    sendCmd2(1)

                    # SAVE Action
                elif JOB_HowTo_Open == True and strDecode == 'stop call back':
                    JJOB = False
                    JOB_HowTo_Open = False
                    STPindex = 0
                    JOB_SAVE = True
                    print "STOP.. Do you want to save? ..."
                    espeak.synth("Do you want to save?")
                    time.sleep(1)
                    talker2("Backward")
                    time.sleep(2)

                elif JOB_SAVE == True and strDecode == 'yes':

                    with sqlite3.connect("Test_PJ2.db") as con:
                        cur2 = con.cursor()
                        cur2.execute('select ID from ActionName where Name = ?', (STPname,))
                        row = cur2.fetchone()
                        for element11 in row:
                            id1 = int(element11)

                            cur3 = con.cursor()
                            cur3.execute('delete from Action_Robot where ID = ?', (id1,))

                    list1 = []
                    for i in select_Buffer():
                        list1.append(selectID_AcName(STPname))
                        for x in i:
                            print(x, "...")
                            list1.append(x)
                        with sqlite3.connect("Test_PJ2.db") as con:
                            cur4 = con.cursor()
                            cur4.execute(
                                'insert into Action_Robot (ID,StepAction,M1,M2,M3,M4,M5,M6,M7,M8) values (?,?,?,?,?,?,?,?,?,?)',
                                (list1))
                            print(list1)
                            del list1[:]

                    del_buff()

                    print "SAVE action !"
                    JOB = True
                    JOB_SAVE = False
                    print "\n------------------------------------------"

                elif JOB_SAVE == True and strDecode == 'no':
                    print "del buff"
                    print "OK No!"
                    espeak.synth("Ok, No")
                    time.sleep(2)
                    del_buff()
                    JOB_SAVE = False
                    JOB = True
                    print "\n------------------------------------------"


                # >>>>>>> PASS DO YOU KNOW~??? <<<<<<<<<<<<
                elif JOB == True and strDecode[:11] == 'do you know':
                    JOB = False
                    print "\n--------------------do you know----------------------"
                    print '\nStream decoding result:', strDecode
                    obj_name = get_object_question(strDecode)
                    print(obj_name)
                    obj_find = search_object_Train(obj_name)
                    status = search_status(obj_name)

                    if obj_find != "None" and status == "yes":

                        print "Yes , I know!"
                        espeak.synth("Yes, I know")
                        time.sleep(4)

                    else:
                        print "No , I don't know!"
                        espeak.synth("No , I don't know!")
                        time.sleep(4)
                    print "\n------------------------------------------"
                    JOB = True

                elif TRAIN_DATA_SET == True and JOB == True and strDecode == "training data set":
                    JOB = False
                    TRAIN_DATA_SET = False
                    print "\n-------------------training data set-----------------------"
                    print "Do you want to save model? (yes or no)"
                    STM = "Do you want to save model?"
                    firebase.post('/robotsend/data', {'textrobot': str(STM)})
                    espeak.synth("Do you want to save model?")
                    time.sleep(5)

                elif TRAIN_DATA_SET == False and strDecode == "yes":
                    print "! training data set"
                    espeak.synth("Just a moment, please.")
                    time.sleep(4)
                    save_model()
                    espeak.synth("OK, finish")
                    time.sleep(3)
                    JOB = True
                    TRAIN_DATA_SET = True
                    print "\n------------------------------------------"

                elif TRAIN_DATA_SET == False and strDecode == "No":
                    print "OK No!"
                    espeak.synth("Ok, No")
                    time.sleep(2)
                    JOB = True
                    TRAIN_DATA_SET = True
                    print "\n------------------------------------------"

    else :
            continue

print "OK"