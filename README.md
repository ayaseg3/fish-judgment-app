# fish-judgment-app

## Overview
The web application just inputs an image of the fish and it will determine the name of the image.

<img width="721" alt="スクリーンショット 2022-04-18 16 38 27" src="https://user-images.githubusercontent.com/75469712/163774089-7dab9148-1695-488b-8337-2cadb58c50c9.png">

The image above is an example of operation.

## Requirement
* numpy 1.19.2
* opencv-python 4.5.1.48
* keras 2.6.0
* Pillow 8.1.0
* Flask 2.1.1
* Werkzeug 2.1.1

## Installation
```bash
pip install numpy==1.19.2
pip install opencv-python==4.5.1.48
pip install keras==2.6.0
pip install pillow==8.1.0
pip install flask==2.1.1
pip install werkzeug==2.1.1
```

## Usage
```bash
git clone https://github.com/ayaseg3/fish-judgment-app.git
cd fish-judgment-app
python app.py
```

## Note
* There is an html file in the ***templates*** folder with the same path.
* ***my_model400.h5*** is a pre-trained model using CNN.
* Since it is in the prototype stage, we are only learning 10 kinds of river fish.
