from firebase import Firebase
import os

config = {
  "apiKey": "AIzaSyB26qPRFCSaIxX2OrSryRpZqPCQ0OJGrUY",
  "authDomain": "fruity501.firebaseapp.com",
  "databaseURL": "https://fruity501-default-rtdb.europe-west1.firebasedatabase.app",
  "projectId": "fruity501",
  "storageBucket": "fruity501.appspot.com",
  "messagingSenderId": "880606982704",
  "appId": "1:880606982704:web:ea8d9fa55ca6563fd1f3f5",
  "measurementId": "G-YCEG67HVLM",
    "serviceAccount" : "serviceAccountKey.json"}
firebase = Firebase(config)
storage = firebase.storage() 

all_files = storage.list_files()

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path[0:-3]+'data\\generated_data\\'

try:
    os.mkdir(dir_path)
except:
    pass

for file in all_files:
    try:
        file.download_to_filename(dir_path + file.name + '.png')
    except:
        os.mkdir(dir_path + file.name[0:-14])
        print('Folder created :' + file.name[0:-14])