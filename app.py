import numpy as np
import cv2
from keras.models import  load_model
from PIL import Image
from flask import Flask, render_template, redirect, request, send_from_directory,jsonify, url_for,send_file, make_response, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path
import os
import base64
from io import BytesIO

app = Flask(__name__)
UPLOAD_FOLDER = './result'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

SearchName = ['アマゴ','イワナ','ウグイ','カジカ','サクラマス','ニジマス','ヒメマス','フナ','ヤマメ','イトウ']
global model
model = load_model('./my_model400.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.' ,1)[1].lower() in ALLOWED_EXTENSIONS

def detect_face(image):
    global name
    global name2
    global name_id
    global name2_id
    trans_pic = background_transparency(image)
    trans_pic = cv2.cvtColor(trans_pic, cv2.COLOR_RGBA2RGB)
    trans_pic = cv2.resize(trans_pic,(64,64)) #(64,64)
    trans_pic = np.expand_dims(trans_pic,axis=0) # [64,64,3]に画像をlistに変更
    name = detect_who(trans_pic)
    name2 = detect_who2(trans_pic)
    print("一番確率が高い魚：",name)
    print("二番目に確率が高い魚：",name2)
    name_id = fish_id(name)
    name2_id = fish_id(name2)
    return image

def detect_who(img):
    global score_1
    global score_2
    name=""
    pre = model.predict(img)
    score_1 = np.max(pre)
    score_2 = sorted(pre.ravel())[-2]
    nameNumLabel=np.argmax(pre)
    if nameNumLabel== 0: 
        name="アマゴ"
    elif nameNumLabel==1:
        name="イワナ"
    elif nameNumLabel==2:
        name="ウグイ"
    elif nameNumLabel==3:
        name="カジカ"
    elif nameNumLabel==4:
        name="サクラマス"
    elif nameNumLabel==5:
        name="ニジマス"
    elif nameNumLabel==6:
        name="ヒメマス" 
    elif nameNumLabel==7:
        name="フナ" 
    elif nameNumLabel==8:
        name="ヤマメ" 
    elif nameNumLabel==9:
        name="イトウ" 
    return name

def detect_who2(img):
    name2=""
    pre = model.predict(img)
    A = np.array(pre) 
    nameNumLabel=A.argsort()[0,8]
    if nameNumLabel== 0: 
        name2="アマゴ"
    elif nameNumLabel==1:
        name2="イワナ"
    elif nameNumLabel==2:
        name2="ウグイ"
    elif nameNumLabel==3:
        name2="カジカ"
    elif nameNumLabel==4:
        name2="サクラマス"
    elif nameNumLabel==5:
        name2="ニジマス"
    elif nameNumLabel==6:
        name2="ヒメマス" 
    elif nameNumLabel==7:
        name2="フナ" 
    elif nameNumLabel==8:
        name2="ヤマメ" 
    elif nameNumLabel==9:
        name2="イトウ" 
    return name2

def fish_id(name):
    name_id = ""
    if name == "アマゴ": 
        name_id = "664"
    elif name == "イワナ":
        name_id = "5486"
    elif name == "ウグイ":
        name_id = "129"
    elif name == "カジカ":
        name_id = "5661"
    elif name == "サクラマス":
        name_id = "268"
    elif name == "ニジマス":
        name_id = "457"
    elif name == "ヒメマス":
        name_id = "139" 
    elif name == "フナ":
        name_id = "6603"
    elif name == "ヤマメ":
        name_id = "161"
    elif name == "イトウ":
        name_id = "656"
    return name_id

def background_transparency(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    ) 
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, contours, -1, color=255, thickness=-1)
    rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)    
    rgba[..., 3] = mask
    for r in range(len(rgba)):
                for c in range(len(rgba[r])):
                    if rgba[r][c][3] == 0:
                        rgba[r][c] = [0, 255, 0, 0]
    return rgba

def savepic(image):
    # 画像書き込み用バッファを確保して画像データをそこに書き込む
    buf = BytesIO()
    image.save(buf,format="png")
    # バイナリデータをbase64でエンコードし、それをさらにutf-8でデコードしておく
    b64str = base64.b64encode(buf.getvalue()).decode("utf-8") 
    # image要素のsrc属性に埋め込めこむために、適切に付帯情報を付与する
    b64data = "data:image/png;base64,{}".format(b64str)
    return b64data

@app.route('/', methods=['GET','POST'])
def uploads_file():
    global orimage
    orimage=""
    res_filename = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'pic01.jpg'))
            image=cv2.imread(UPLOAD_FOLDER + "/pic01.jpg")
            if image is None:
                print("Not open:")
                return redirect(request.url)
            b,g,r = cv2.split(image)
            image = cv2.merge([r,g,b])
            imagee = Image.fromarray(image)
            orimage = savepic(imagee)
            whoImage=detect_face(image)
            pil_img = Image.fromarray(whoImage)
            global b64data1
            b64data1 = savepic(pil_img)
    return render_template('index.html',\
        originalimage = orimage)

@app.route('/result', methods=['GET','POST'])
def result_file():
    return render_template('index2.html',\
        name = name,\
        score_1 = int(100*score_1),\
        score_2 = int(100*score_2),\
        name2 = name2,\
        name_id = "https://zukan.com/fish/internal" + name_id,\
        name2_id = "https://zukan.com/fish/internal" + name2_id,\
        b64data = b64data1)

if __name__ == '__main__':
    app.debug = True
    app.run(host='localhost', port=3000)