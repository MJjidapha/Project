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


def callback_sensor(data):
    #print(data.data)
    A = data.data
    print A[0]
    #time.sleep(1)
    from firebase import firebase
    firebase = firebase.FirebaseApplication('https://dogwood-terra-184417.firebaseio.com/', None)
    result = firebase.get('/Gauges/data/',None)
    if(result != None):
        result = firebase.delete('/Gauges/data/',None)
    firebase.post('/Gauges/data/',{'Battery':int(A[1]),'temperature':int(A[0])})
# read 2 sensors (temperature, battery)
def readSensor(msg): # send msg with 2
    pub1 = rospy.Publisher("command", UInt8, queue_size=10)
    #rospy.Subscriber("sensors", UInt16MultiArray, callback_sensor)
    rospy.init_node('Move', anonymous=True)
    pub1.publish(int(msg))





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
    pathPicCom2 = "/home/kns/PycharmProjects/Aj/AJ2/pic/" + objectFirebese + "/" + objectFirebese + num + "_2.jpg"
    pathPicCom3 = "/home/kns/PycharmProjects/Aj/AJ2/pic/" + objectFirebese + "/" + objectFirebese + num + "_3.jpg"
    pathPicCom4 = "/home/kns/PycharmProjects/Aj/AJ2/pic/" + objectFirebese + "/" + objectFirebese + num + "_4.jpg"
    pathPicCom5 = "/home/kns/PycharmProjects/Aj/AJ2/pic/" + objectFirebese + "/" + objectFirebese + num + "_5.jpg"
    pathPicCom6 = "/home/kns/PycharmProjects/Aj/AJ2/pic/" + objectFirebese + "/" + objectFirebese + num + "_6.jpg"
    pathPicCom7 = "/home/kns/PycharmProjects/Aj/AJ2/pic/" + objectFirebese + "/" + objectFirebese + num + "_7.jpg"
    pathPicCom8 = "/home/kns/PycharmProjects/Aj/AJ2/pic/" + objectFirebese + "/" + objectFirebese + num + "_8.jpg"
    pathPicCom9 = "/home/kns/PycharmProjects/Aj/AJ2/pic/" + objectFirebese + "/" + objectFirebese + num + "_9.jpg"
    pathPicCom10 = "/home/kns/PycharmProjects/Aj/AJ2/pic/" + objectFirebese + "/" + objectFirebese + num + "_10.jpg"
    print(pathPicCom)
    blob1 = bucket.blob(objectFirebese+ num + "_1.jpg")
    blob2 = bucket.blob(objectFirebese + num + "_2.jpg")
    blob3 = bucket.blob(objectFirebese + num + "_3.jpg")
    blob4 = bucket.blob(objectFirebese + num + "_4.jpg")
    blob5 = bucket.blob(objectFirebese + num + "_5.jpg")
    blob6 = bucket.blob(objectFirebese + num + "_6.jpg")
    blob7 = bucket.blob(objectFirebese + num + "_7.jpg")
    blob8 = bucket.blob(objectFirebese + num + "_8.jpg")
    blob9 = bucket.blob(objectFirebese + num + "_9.jpg")
    blob10 = bucket.blob(objectFirebese + num + "_10.jpg")
    blob1.upload_from_filename(filename=pathPicCom)
    blob2.upload_from_filename(filename=pathPicCom2)
    blob3.upload_from_filename(filename=pathPicCom3)
    blob4.upload_from_filename(filename=pathPicCom4)
    blob5.upload_from_filename(filename=pathPicCom5)
    blob6.upload_from_filename(filename=pathPicCom6)
    blob7.upload_from_filename(filename=pathPicCom7)
    blob8.upload_from_filename(filename=pathPicCom8)
    blob9.upload_from_filename(filename=pathPicCom9)
    blob10.upload_from_filename(filename=pathPicCom10)
    firebase.post('/object/data', {'object_Name': objectFirebese,
                                   'thumbnail': 'https://firebasestorage.googleapis.com/v0/b/dogwood-terra-184417.appspot.com/o/' + objectFirebese +num+ '_1.jpg' + '?alt=media'
                                    ,'thumbnail2': 'https://firebasestorage.googleapis.com/v0/b/dogwood-terra-184417.appspot.com/o/' + objectFirebese +num+ '_2.jpg' + '?alt=media'
                                    ,'thumbnail3': 'https://firebasestorage.googleapis.com/v0/b/dogwood-terra-184417.appspot.com/o/' + objectFirebese +num+ '_3.jpg' + '?alt=media'
                                    ,'thumbnail4': 'https://firebasestorage.googleapis.com/v0/b/dogwood-terra-184417.appspot.com/o/' + objectFirebese +num+ '_4.jpg' + '?alt=media'
                                    ,'thumbnail5': 'https://firebasestorage.googleapis.com/v0/b/dogwood-terra-184417.appspot.com/o/' + objectFirebese + num + '_5.jpg' + '?alt=media'
                                    ,'thumbnail6': 'https://firebasestorage.googleapis.com/v0/b/dogwood-terra-184417.appspot.com/o/' + objectFirebese + num + '_6.jpg' + '?alt=media'
                                    ,'thumbnail7': 'https://firebasestorage.googleapis.com/v0/b/dogwood-terra-184417.appspot.com/o/' + objectFirebese + num + '_7.jpg' + '?alt=media'
                                    ,'thumbnail8': 'https://firebasestorage.googleapis.com/v0/b/dogwood-terra-184417.appspot.com/o/' + objectFirebese + num + '_8.jpg' + '?alt=media'
                                    ,'thumbnail9': 'https://firebasestorage.googleapis.com/v0/b/dogwood-terra-184417.appspot.com/o/' + objectFirebese + num + '_9.jpg' + '?alt=media'
                                    ,'thumbnail10': 'https://firebasestorage.googleapis.com/v0/b/dogwood-terra-184417.appspot.com/o/' + objectFirebese + num + '_10.jpg' + '?alt=media'})


def update():
    #############pictureupdate###########

    from firebase import firebase
    from google.cloud import storage
    client = storage.Client.from_service_account_json(
        'C:\Users\MOJI\Desktop\project\dogwood-terra-184417-firebase-adminsdk-vy9o9-765ac92f9f.json')
    bucket = client.get_bucket('dogwood-terra-184417.appspot.com')
    firebase = firebase.FirebaseApplication('https://dogwood-terra-184417.firebaseio.com/', None)
    import requests
    import os
    def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print ('Error: Creating directory. ' + directory)

    import sqlite3
    conn = sqlite3.connect('C:\Users\MOJI\Desktop\project\Firebase\python\Test_PJ2.db')
    cursor = conn.execute("SELECT ID,Name,day,month,year,hour,minute from call_Detect")
    result = firebase.get('/object/data/', None)
    n = []
    Na = []
    sameDay = []
    sameMonth = []
    sameYear = []
    sameHour = []
    sameMinute = []
    id2 = []
    Maxid = 0

    for row in cursor:
        id2.append(row[0])
        Na.append(row[1])
        sameDay.append(row[2])
        sameMonth.append(row[3])
        sameYear.append(row[4])
        sameHour.append(row[5])
        sameMinute.append(row[6])
        if (row[0] > Maxid):
            Maxid = row[0]
    cursor.close()
    # print NaM
    # print sameY
    for i in result:
        name = firebase.get('/object/data/' + i, 'object_Name')
        n.append(name)

    # print n
    # print Na
    Maxid = Maxid + 1
    nobcheck = set(n).difference(Na)
    nobcheck = list(nobcheck)
    ln = len(nobcheck)
    # print nobcheck
    nobcheck2 = set(Na).difference(n)
    nobcheck2 = list(nobcheck2)
    ln2 = len(nobcheck2)
    # print nobcheck2
    ln6 = len(Na)
    count = 0
    count1 = 0
    count2 = 0

    for j in range(ln6):
        for i in result:
            resultchek = firebase.get('/object/data/' + i, None)
            listre = list(resultchek)
            lnre = len(listre) - 2
            if (resultchek != None):
                name = firebase.get('/object/data/' + i, 'object_Name')
                date = firebase.get('/object/data/' + i, 'date')
                listdate = date.split(",")
                # print(type(listdate[0]))
                if (count1 < ln6):
                    if (name == Na[count1]):
                        print "..................."
                        print name
                        print sameYear[count1]
                        print Na[count1]
                        s = 0
                        f = 0
                        yearF = int(listdate[2])
                        monthF = int(listdate[1])
                        dayF = int(listdate[0])
                        hourF = int(listdate[3])
                        minuteF = int(listdate[4])
                        if (yearF == sameYear[count1]):
                            if (monthF == sameMonth[count1]):
                                if (dayF == sameDay[count1]):
                                    if (hourF == sameHour[count1]):
                                        if (minuteF == sameMinute[count1]):
                                            print 'equal'
                                        elif (minuteF > sameMinute[count1]):
                                            print "dayF"
                                            f = 1
                                        else:
                                            s = 1
                                            print "minuteS"
                                    elif (hourF > sameHour[count1]):
                                        print "hourF"
                                        f = 1
                                    else:
                                        s = 1
                                        print "hourS"
                                elif (dayF > sameDay[count1]):
                                    print "dayF"
                                    f = 1
                                else:
                                    s = 1
                                    print "dayS"
                            elif (monthF > sameMonth[count1]):
                                print "monthF"
                                f = 1
                            else:
                                s = 1
                                # print (monthS)
                        elif (yearF > sameYear[count1]):
                            print "yearF"
                            f = 1
                        else:
                            s = 1
                            print "yearS"
                        if (s == 1):
                            print 's'
                            print sameDay[count1], sameMonth[count1], sameYear[count1], sameHour[count1], sameMinute[
                                count1]
                            fileaddress = ("C:\Users\MOJI\Pictures\\") + str(row[1])
                            print fileaddress
                            filename1 = str(row[1]) + '1_1.jpg'
                            blob1 = bucket.blob(str(row[1]) + '1_1.jpg')
                            blob1.upload_from_filename(filename=fileaddress + "\\" + filename1)
                            filename2 = str(row[1]) + '1_2.jpg'
                            blob2 = bucket.blob(str(row[1]) + '1_2.jpg')
                            blob2.upload_from_filename(filename=fileaddress + "\\" + filename2)
                            filename3 = str(row[1]) + '1_3.jpg'
                            blob3 = bucket.blob(str(row[1]) + '1_3.jpg')
                            blob3.upload_from_filename(filename=fileaddress + "\\" + filename3)
                            filename4 = str(row[1]) + '1_4.jpg'
                            blob4 = bucket.blob(str(row[1]) + '1_4.jpg')
                            blob4.upload_from_filename(filename=fileaddress + "\\" + filename4)
                            filename5 = str(row[1]) + '1_5.jpg'
                            blob5 = bucket.blob(str(row[1]) + '1_5.jpg')
                            blob5.upload_from_filename(filename=fileaddress + "\\" + filename5)
                            filename6 = str(row[1]) + '1_6.jpg'
                            blob6 = bucket.blob(str(row[1]) + '1_6.jpg')
                            blob6.upload_from_filename(filename=fileaddress + "\\" + filename6)
                            filename7 = str(row[1]) + '1_7.jpg'
                            blob7 = bucket.blob(str(row[1]) + '1_7.jpg')
                            blob7.upload_from_filename(filename=fileaddress + "\\" + filename7)
                            filename8 = str(row[1]) + '1_8.jpg'
                            blob8 = bucket.blob(str(row[1]) + '1_8.jpg')
                            blob8.upload_from_filename(filename=fileaddress + "\\" + filename8)
                            filename9 = str(row[1]) + '1_9.jpg'
                            blob9 = bucket.blob(str(row[1]) + '1_9.jpg')
                            blob9.upload_from_filename(filename=fileaddress + "\\" + filename9)
                            filename10 = str(row[1]) + '1_10.jpg'
                            blob10 = bucket.blob(str(row[1]) + '1_10.jpg')
                            blob10.upload_from_filename(filename=fileaddress + "\\" + filename10)
                            result4 = firebase.delete('/object/data/' + i, None)
                            firebase.post('/object/data', {'object_Name': str(row[1]),
                                                           'thumbnail': 'https://firebasestorage.googleapis.com/v0/b/dogwood-terra-184417.appspot.com/o/' + filename1 + '?alt=media'
                                ,
                                                           'thumbnail2': 'https://firebasestorage.googleapis.com/v0/b/dogwood-terra-184417.appspot.com/o/' + filename2 + '?alt=media'
                                ,
                                                           'thumbnail3': 'https://firebasestorage.googleapis.com/v0/b/dogwood-terra-184417.appspot.com/o/' + filename3 + '?alt=media'
                                ,
                                                           'thumbnail4': 'https://firebasestorage.googleapis.com/v0/b/dogwood-terra-184417.appspot.com/o/' + filename4 + '?alt=media'
                                ,
                                                           'thumbnail5': 'https://firebasestorage.googleapis.com/v0/b/dogwood-terra-184417.appspot.com/o/' + filename5 + '?alt=media'
                                ,
                                                           'thumbnail6': 'https://firebasestorage.googleapis.com/v0/b/dogwood-terra-184417.appspot.com/o/' + filename6 + '?alt=media'
                                ,
                                                           'thumbnail7': 'https://firebasestorage.googleapis.com/v0/b/dogwood-terra-184417.appspot.com/o/' + filename7 + '?alt=media'
                                ,
                                                           'thumbnail8': 'https://firebasestorage.googleapis.com/v0/b/dogwood-terra-184417.appspot.com/o/' + filename8 + '?alt=media'
                                ,
                                                           'thumbnail9': 'https://firebasestorage.googleapis.com/v0/b/dogwood-terra-184417.appspot.com/o/' + filename9 + '?alt=media'
                                ,
                                                           'thumbnail10': 'https://firebasestorage.googleapis.com/v0/b/dogwood-terra-184417.appspot.com/o/' + filename10 + '?alt=media'
                                , 'date': str(sameDay[count1]) + ',' + str(sameMonth[count1]) + ',' + str(
                                    sameYear[count1]) + ',' + str(sameHour[count1]) + ',' + str(sameMinute[count1])})
                        elif (f == 1):
                            print 'f'
                            conn = sqlite3.connect('C:\Users\MOJI\Desktop\project\Firebase\python\Test_PJ2.db')
                            conn.execute("DELETE from call_Detect where ID = " + str(id2[count1]) + ";")
                            conn.commit()
                            with sqlite3.connect('C:\Users\MOJI\Desktop\project\Firebase\python\Test_PJ2.db') as con:
                                cur3 = con.cursor()
                                cur3.execute('insert into call_Detect values (?,?,?,?,?,?,?,?)', (
                                id2[count1], name, 1, int(listdate[0]), int(listdate[1]), int(listdate[2]),
                                int(listdate[3]), int(listdate[4])))
                            for k in range(lnre):
                                print k
                                if (k == 0):
                                    url = firebase.get('/object/data/' + i, 'thumbnail')
                                    print url, '111'
                                else:
                                    url = firebase.get('/object/data/' + i, 'thumbnail' + str(k + 1))
                                    print url
                                r = requests.get(url)
                                folder = createFolder('C:\Users\MOJI\Pictures./' + name + '/')
                                with open('C:\\Users\\MOJI\\Pictures\\' + name + '\\' + name + str(k + 1) + '.jpg',
                                          'wb') as f:
                                    f.write(r.content)
        count1 += 1

    result = firebase.get('/object/data/', None)
    for j in range(ln):
        for i in result:
            resultchek = firebase.get('/object/data/' + i, None)
            listre = list(resultchek)
            lnre = len(listre) - 2
            if (resultchek != None):
                name = firebase.get('/object/data/' + i, 'object_Name')
                date = firebase.get('/object/data/' + i, 'date')
                listdate = date.split(",")
                if (count < ln):
                    if (name == nobcheck[count]):
                        print name, Maxid, int(listdate[0]), int(listdate[1]), int(listdate[2]), int(listdate[3]), int(
                            listdate[4])
                        with sqlite3.connect('C:\Users\MOJI\Desktop\project\Firebase\python\Test_PJ2.db') as con:
                            cur3 = con.cursor()
                            cur3.execute('insert into call_Detect values (?,?,?,?,?,?,?,?)', (
                            Maxid, name, 1, int(listdate[0]), int(listdate[1]), int(listdate[2]), int(listdate[3]),
                            int(listdate[4])))
                        # print lnre
                        for k in range(lnre):
                            print k
                            if (k == 0):
                                url = firebase.get('/object/data/' + i, 'thumbnail')
                                print url, '111'
                            else:
                                url = firebase.get('/object/data/' + i, 'thumbnail' + str(k + 1))
                                print url
                            r = requests.get(url)
                            folder = createFolder('C:\Users\MOJI\Downloads./' + name + '/')
                            with open('C:\\Users\\MOJI\\Downloads\\' + name + '\\' + name + str(k + 1) + '_1' + '.jpg',
                                      'wb') as f:
                                f.write(r.content)
        count += 1
        Maxid += 1

    cursor = conn.execute("SELECT ID, Name,K,day,month,year,hour,minute from call_Detect")
    for j in range(ln2):
        for row in cursor:
            if (count2 < ln2):
                if (row[1] == nobcheck2[count2]):
                    print row[1]
                    day = row[3]
                    month = row[4]
                    year = row[5]
                    hour = row[6]
                    minute = row[7]
                    print day, month, year, hour, minute
                    fileaddress = ("C:\Users\MOJI\Pictures\\") + str(row[1])
                    print fileaddress
                    filename1 = str(row[1]) + '1.jpg'
                    blob1 = bucket.blob(str(row[1]) + '1.jpg')
                    blob1.upload_from_filename(filename=fileaddress + "\\" + filename1)
                    filename2 = str(row[1]) + '2.jpg'
                    blob2 = bucket.blob(str(row[1]) + '2.jpg')
                    blob2.upload_from_filename(filename=fileaddress + "\\" + filename2)
                    filename3 = str(row[1]) + '3.jpg'
                    blob3 = bucket.blob(str(row[1]) + '3.jpg')
                    blob3.upload_from_filename(filename=fileaddress + "\\" + filename3)
                    filename4 = str(row[1]) + '4.jpg'
                    blob4 = bucket.blob(str(row[1]) + '4.jpg')
                    blob4.upload_from_filename(filename=fileaddress + "\\" + filename4)
                    filename5 = str(row[1]) + '5.jpg'
                    blob5 = bucket.blob(str(row[1]) + '5.jpg')
                    blob5.upload_from_filename(filename=fileaddress + "\\" + filename5)
                    filename6 = str(row[1]) + '6.jpg'
                    blob6 = bucket.blob(str(row[1]) + '6.jpg')
                    blob6.upload_from_filename(filename=fileaddress + "\\" + filename6)
                    filename7 = str(row[1]) + '7.jpg'
                    blob7 = bucket.blob(str(row[1]) + '7.jpg')
                    blob7.upload_from_filename(filename=fileaddress + "\\" + filename7)
                    filename8 = str(row[1]) + '8.jpg'
                    blob8 = bucket.blob(str(row[1]) + '8.jpg')
                    blob8.upload_from_filename(filename=fileaddress + "\\" + filename8)
                    filename9 = str(row[1]) + '9.jpg'
                    blob9 = bucket.blob(str(row[1]) + '9.jpg')
                    blob9.upload_from_filename(filename=fileaddress + "\\" + filename9)
                    filename10 = str(row[1]) + '10.jpg'
                    blob10 = bucket.blob(str(row[1]) + '10.jpg')
                    blob10.upload_from_filename(filename=fileaddress + "\\" + filename10)
                    firebase.post('/object/data', {'object_Name': str(row[1]),
                                                   'thumbnail': 'https://firebasestorage.googleapis.com/v0/b/dogwood-terra-184417.appspot.com/o/' + filename1 + '?alt=media'
                                                    ,'thumbnail2': 'https://firebasestorage.googleapis.com/v0/b/dogwood-terra-184417.appspot.com/o/' + filename2 + '?alt=media'
                                                    ,'thumbnail3': 'https://firebasestorage.googleapis.com/v0/b/dogwood-terra-184417.appspot.com/o/' + filename3 + '?alt=media'
                                                    ,'thumbnail4': 'https://firebasestorage.googleapis.com/v0/b/dogwood-terra-184417.appspot.com/o/' + filename4 + '?alt=media'
                                                    ,'thumbnail5': 'https://firebasestorage.googleapis.com/v0/b/dogwood-terra-184417.appspot.com/o/' + filename5 + '?alt=media'
                                                    ,'thumbnail6': 'https://firebasestorage.googleapis.com/v0/b/dogwood-terra-184417.appspot.com/o/' + filename6 + '?alt=media'
                                                    ,'thumbnail7': 'https://firebasestorage.googleapis.com/v0/b/dogwood-terra-184417.appspot.com/o/' + filename7 + '?alt=media'
                                                    ,'thumbnail8': 'https://firebasestorage.googleapis.com/v0/b/dogwood-terra-184417.appspot.com/o/' + filename8 + '?alt=media'
                                                    ,'thumbnail9': 'https://firebasestorage.googleapis.com/v0/b/dogwood-terra-184417.appspot.com/o/' + filename9 + '?alt=media'
                                                    ,'thumbnail10': 'https://firebasestorage.googleapis.com/v0/b/dogwood-terra-184417.appspot.com/o/' + filename10 + '?alt=media'
                                                    ,'date': str(day) + ',' + str(month) + ',' + str(year) + ',' + str(hour) + ',' + str(minute)})
        count2 += 1
    cursor.close()

    ###################### motorupdate #################

    from firebase import firebase
    from google.cloud import storage
    client = storage.Client.from_service_account_json(
        'C:\Users\MOJI\Desktop\project\dogwood-terra-184417-firebase-adminsdk-vy9o9-765ac92f9f.json')
    bucket = client.get_bucket('dogwood-terra-184417.appspot.com')
    firebase = firebase.FirebaseApplication('https://dogwood-terra-184417.firebaseio.com/', None)
    import sqlite3
    conn = sqlite3.connect('C:\Users\MOJI\Desktop\project\Firebase\python\Test_PJ2.db')
    cursor = conn.execute("SELECT ID,Name,day,month,year,hour,minute from ActionName")
    result = firebase.get('/action/data/', None)
    nM = []
    NaM = []
    sameD = []
    sameM = []
    sameY = []
    sameH = []
    sameMi = []
    id1 = []
    maxid = 0
    for row in cursor:
        id1.append(row[0])
        NaM.append(row[1])
        sameD.append(row[2])
        sameM.append(row[3])
        sameY.append(row[4])
        sameH.append(row[5])
        sameMi.append(row[6])
        if (row[0] > maxid):
            maxid = row[0]
    cursor.close()

    # print NaM
    # print sameY
    for i in result:
        name = firebase.get('/action/data/' + i, 'action_name')
        nM.append(name)
    # print sameNM,sameDt
    # print lnsameNM
    # print n
    # print Na
    # print maxid
    maxid = maxid + 1
    nMocheck = set(nM).difference(NaM)
    nMocheck = list(nMocheck)
    ln3 = len(nMocheck)
    # print ln3
    # print nMocheck
    nMocheck2 = set(NaM).difference(nM)
    nMocheck2 = list(nMocheck2)
    ln4 = len(nMocheck2)
    # print ln4,nMocheck2
    ln5 = len(NaM)
    count3 = 0
    count4 = 0
    count5 = 0

    for j in range(ln5):
        for i in result:
            resultchek = firebase.get('/action/data/' + i, None)
            if (resultchek != None):
                name = firebase.get('/action/data/' + i, 'action_name')
                motor = firebase.get('/action/data/' + i, 'motor')
                motor = str(motor)
                motor = motor.split()
                listmotor = motor[1].split(",")
                date = firebase.get('/action/data/' + i, 'date')
                listdate = date.split(",")
                # print(type(listdate[0]))
                step = firebase.get('/action/data/' + i, 'step')
                if (count5 < ln5):
                    if (name == NaM[count5]):
                        print "..................."
                        print name
                        print sameY[count5]
                        print NaM[count5]
                        s = 0
                        f = 0
                        yearF = int(listdate[2])
                        monthF = int(listdate[1])
                        dayF = int(listdate[0])
                        hourF = int(listdate[3])
                        minuteF = int(listdate[4])
                        if (yearF == sameY[count5]):
                            if (monthF == sameM[count5]):
                                if (dayF == sameD[count5]):
                                    if (hourF == sameH[count5]):
                                        if (minuteF == sameMi[count5]):
                                            print 'equal'
                                        elif (minuteF > sameMi[count5]):
                                            print "dayF"
                                            f = step
                                        else:
                                            s = step
                                            print "minuteS"
                                    elif (hourF > sameH[count5]):
                                        print "hourF"
                                        f = step
                                    else:
                                        s = step
                                        print "hourS"
                                elif (dayF > sameD[count5]):
                                    print "dayF"
                                    f = step
                                else:
                                    s = step
                                    print "dayS"
                            elif (monthF > sameM[count5]):
                                print "monthF"
                                f = step
                            else:
                                s = step
                                # print (monthS)
                        elif (yearF > sameY[count5]):
                            print "yearF"
                            f = step
                        else:
                            s = step
                            print "yearS"
                        if (step == s):
                            for j in result:
                                result3 = firebase.get('/action/data/' + j, 'action_name')
                                if (str(result3) == str(NaM[count5])):
                                    result4 = firebase.delete('/action/data/' + j, None)
                            cursor3 = conn.execute("SELECT ID, stepAction,M1,M2,M3,M4,M5,M6,M7,M8 from Action_Robot")
                            for row3 in cursor3:
                                if (id1[count5] == row3[0]):
                                    firebase.post('/action/data', {'action_name': NaM[count5], 'step': row3[1],
                                                                   'motor': "[ " + str(row3[2]) + "," + str(
                                                                       row3[3]) + "," + str(row3[4]) + "," + str(
                                                                       row3[5]) + "," + str(row3[6]) + "," + str(
                                                                       row3[7]) + "," + str(row3[8]) + "," + str(
                                                                       row3[9]) + " ]",
                                                                   'date': str(row[2]) + "," + str(row[3]) + "," + str(
                                                                       row[4]) + "," + str(row[5]) + "," + str(row[6])})
                            cursor3.close()
                        elif (step == f):
                            print name, "aaaa", step, 'bbb', id1[count5]
                            conn = sqlite3.connect('C:\Users\MOJI\Desktop\project\Firebase\python\Test_PJ2.db')
                            conn.execute("DELETE from ActionName where ID = " + str(id1[count5]) + ";")
                            conn.commit()
                            conn = sqlite3.connect('C:\Users\MOJI\Desktop\project\Firebase\python\Test_PJ2.db')
                            conn.execute("DELETE from Action_Robot where ID = " + str(id1[count5]) + ";")
                            conn.commit()
                            if (str(step) == '1'):
                                with sqlite3.connect(
                                        'C:\Users\MOJI\Desktop\project\Firebase\python\Test_PJ2.db') as con:
                                    cur3 = con.cursor()
                                    cur3.execute('insert into ActionName values (?,?,?,?,?,?,?)', (
                                    int(id1[count5]), str(name), int(listdate[0]), int(listdate[1]), int(listdate[2]),
                                    int(listdate[3]), int(listdate[4])))

                            with sqlite3.connect('C:\Users\MOJI\Desktop\project\Firebase\python\Test_PJ2.db') as con:
                                cur4 = con.cursor()
                                cur4.execute('insert into Action_Robot values (?,?,?,?,?,?,?,?,?,?)', (
                                int(id1[count5]), int(step), int(listmotor[0]), int(listmotor[1]), int(listmotor[2]),
                                int(listmotor[3]), int(listmotor[4]), int(listmotor[5]), int(listmotor[6]),
                                int(listmotor[7])))
        count5 += 1

    result = firebase.get('/action/data/', None)
    for j in range(ln3):
        for i in result:
            resultchek = firebase.get('/action/data/' + i, None)
            if (resultchek != None):
                name = firebase.get('/action/data/' + i, 'action_name')
                motor = firebase.get('/action/data/' + i, 'motor')
                motor = str(motor)
                motor = motor.split()
                listmotor = motor[1].split(",")
                date = firebase.get('/action/data/' + i, 'date')
                listdate = date.split(",")
                # print(type(listdate[0]))
                step = firebase.get('/action/data/' + i, 'step')
                # print name
                if (count3 < ln3):
                    if (name == nMocheck[count3]):
                        print name, listdate, count3, maxid
                        if (str(step) == '1'):
                            with sqlite3.connect('C:\Users\MOJI\Desktop\project\Firebase\python\Test_PJ2.db') as con:
                                cur3 = con.cursor()
                                cur3.execute('insert into ActionName values (?,?,?,?,?,?,?)', (
                                maxid, str(name), int(listdate[0]), int(listdate[1]), int(listdate[2]),
                                int(listdate[3]), int(listdate[4])))

                        with sqlite3.connect('C:\Users\MOJI\Desktop\project\Firebase\python\Test_PJ2.db') as con:
                            cur4 = con.cursor()
                            cur4.execute('insert into Action_Robot values (?,?,?,?,?,?,?,?,?,?)', (
                            maxid, int(step), int(listmotor[0]), int(listmotor[1]), int(listmotor[2]),
                            int(listmotor[3]), int(listmotor[4]), int(listmotor[5]), int(listmotor[6]),
                            int(listmotor[7])))
        count3 += 1
        maxid + 1

    cursor = conn.execute("SELECT ID, Name,Day,Month,Year,Hour,Minute from ActionName")
    for j in range(ln4):
        for row in cursor:
            if (count4 < ln4):
                if (row[1] == nMocheck2[count4]):
                    print row[1]
                    day = row[2]
                    month = row[3]
                    year = row[4]
                    hour = row[5]
                    minute = row[6]
                    print day, month, year, hour, minute
                    cursor2 = conn.execute("SELECT ID, stepAction,M1,M2,M3,M4,M5,M6,M7,M8 from Action_Robot")
                    for row2 in cursor2:
                        if (row[0] == row2[0]):
                            firebase.post('/action/data', {'action_name': row[1], 'step': row2[1],
                                                           'motor': "[ " + str(row2[2]) + "," + str(
                                                               row2[3]) + "," + str(row2[4]) + "," + str(
                                                               row2[5]) + "," + str(row2[6]) + "," + str(
                                                               row2[7]) + "," + str(row2[8]) + "," + str(
                                                               row2[9]) + " ]",
                                                           'date': str(row[2]) + "," + str(row[3]) + "," + str(
                                                               row[4]) + "," + str(row[5]) + "," + str(row[6])})
                    cursor2.close()
        count4 += 1
    cursor.close()
    result2 = firebase.get('/updatecheck/data/', None)
    if result2 != None:
        result2 = firebase.delete('/updatecheck/data/', None)
    firebase.post('/updatecheck/data', {'check': "information match"})
    firebase.post('/robotsend/data', {'textrobot': "Update completed."})

def checkupdate():
    from firebase import firebase
    import sqlite3
    firebase = firebase.FirebaseApplication('https://dogwood-terra-184417.firebaseio.com/', None)
    conn = sqlite3.connect('C:\Users\MOJI\Desktop\project\Firebase\python\Test_PJ2.db')
    cursor = conn.execute("SELECT ID,Name,day,month,year,hour,minute from ActionName")
    result = firebase.get('/action/data/', None)
    result1 = firebase.get('/object/data/', None)
    n = []
    Na = []
    nM = []
    NaM = []
    for row in cursor:
        NaM.append(row[1])
    cursor.close()
    for i in result:
        name = firebase.get('/action/data/' + i, 'action_name')
        nM.append(name)
    nMocheck = set(nM).difference(NaM)
    nMocheck = list(nMocheck)
    ln3 = len(nMocheck)
    nMocheck2 = set(NaM).difference(nM)
    nMocheck2 = list(nMocheck2)
    ln4 = len(nMocheck2)
    cursor = conn.execute("SELECT ID,Name,day,month,year,hour,minute from call_Detect")
    for row in cursor:
        Na.append(row[1])
    cursor.close()

    for i in result1:
        name = firebase.get('/object/data/' + i, 'object_Name')
        n.append(name)

    nobcheck = set(n).difference(Na)
    nobcheck = list(nobcheck)
    ln = len(nobcheck)
    nobcheck2 = set(Na).difference(n)
    nobcheck2 = list(nobcheck2)
    ln2 = len(nobcheck2)
    picture = ln3 + ln4
    motor = ln + ln2
    # print motor,picture
    if (motor != 0 and picture != 0):
        result2 = firebase.get('/updatecheck/data/', None)
        if result2 != None:
            result2 = firebase.delete('/updatecheck/data/', None)
        firebase.post('/updatecheck/data', {
            'check': "Object information does not match and Action information does not match.Please press update button."})
    elif (motor != 0):
        result2 = firebase.get('/updatecheck/data/', None)
        if result2 != None:
            result2 = firebase.delete('/updatecheck/data/', None)
        firebase.post('/updatecheck/data', {'check': "Action information does not match.Please press update button."})
    elif (picture != 0):
        result2 = firebase.get('/updatecheck/data/', None)
        if result2 != None:
            result2 = firebase.delete('/updatecheck/data/', None)
        firebase.post('/updatecheck/data', {'check': "Object information does not match.Please press update button."})
    else:
        result2 = firebase.get('/updatecheck/data/', None)
        if result2 != None:
            result2 = firebase.delete('/updatecheck/data/', None)
    firebase.post('/updatecheck/data', {'check': "information match"})

##############################################################################################################
##############################################################################################################


checkupdate()

from firebase import firebase
firebase = firebase.FirebaseApplication('https://dogwood-terra-184417.firebaseio.com/', None)
CountSensor =  0
rospy.Subscriber("sensors", UInt16MultiArray, callback_sensor)
while (True):
    CountSensor+=1
    if(CountSensor%10==0):
        readSensor(2)

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

                if strDecode == 'update':
                    update()

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
                        firebase.post('/robotsend/data', {'textrobot': str(st)})
                        espeak.synth(st)
                        time.sleep(1)

                    else:

                        keep_First_Home()

                        insert_name()

                        print " check=0 , please say next. If you want to teach me"
                        espeak.synth("OK , please say next. If you want to teach me")
                        time.sleep(2)
                        JOB_HowTo_Open = True


                elif check_Go == True and strDecode == "yes":

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
                elif JOB_HowTo_Open == True and strDecode == 'stop':
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
                        list1.append(selectID_AcName(STPname))  #id
                        for x in i:
                            print(x, "...")
                            list1.append(x)    #step,m1,...,m8
                        with sqlite3.connect("Test_PJ2.db") as con:
                            cur4 = con.cursor()
                            cur4.execute(
                                'insert into Action_Robot (ID,StepAction,M1,M2,M3,M4,M5,M6,M7,M8) values (?,?,?,?,?,?,?,?,?,?)',
                                (list1))
                            firebase.post('/action/data',
                                          {'action_name': str(STPname), 'step': int(list1[1]),
                                           'motor': '[ ' + str(list1[2]) + ',' + str(list1[3]) + ',' + str(
                                               list1[4]) + ',' + str(list1[5]) + ',' + str(list1[6]) + ',' + str(
                                               list1[7]) + ',' + str(list1[8]) + ',' + str(list1[9]) + ' ]'})
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
                        firebase.post('/robotsend/data', {'textrobot': ("Yes, I know")})
                        time.sleep(4)

                    else:
                        print "No , I don't know!"
                        espeak.synth("No , I don't know!")
                        firebase.post('/robotsend/data', {'textrobot': ("No , I don't know!")})
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