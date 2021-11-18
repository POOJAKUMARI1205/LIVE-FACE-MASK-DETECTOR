
<h1 align="center">Proper Mask Wearing Detection and Alarm System</h1>

<div align="center">
    <strong>A lightweight face mask detection that is easy to deploy</strong>
</div>

<div align="center">
    Trained on Tensorflow/Keras. Deployed using Dash on Google App Engine. 
</div>

<br/>

<div align="center">
    <!-- Python version -->
    <img src="https://img.shields.io/badge/python-v3.8-blue?style=flat-square"/>
    <!-- Last commit -->
    <img src="https://img.shields.io/github/last-commit/achen353/Face-Mask-Detector?style=flat-square"/>
    <!-- Stars -->
    <img src="https://img.shields.io/github/stars/achen353/Face-Mask-Detector?style=flat-square"/>
    <!-- Forks -->
    <img src="https://img.shields.io/github/forks/achen353/Face-Mask-Detector?style=flat-square"/>
    <!-- Open Issues -->
    <img src="https://img.shields.io/github/issues/achen353/Face-Mask-Detector?style=flat-square"/>
</div>

<br/>

<div align="center">
    <img src="./readme_assets/readme_cover.png"/>
</div>

<br/>

*Read this in [繁體中文](README.zh-tw.md).*

## Table of Contents
- [Features](#features)
- [About](#about)
- [Frameworks and Libraries](#frameworkslibraries)
- [Datasets](#datasets)
- [Training Results](#training-results)
- [Requirements](#requirements)
- [Setup](#setup) 
- [How to Run](#how-to-run)
- [Dash App Demo](#dash-app-demo)
- [Credits](#credits)
- [License](#license)

## Features
- __Lightweight models:__  only `2,422,339` and `2,422,210` parameters for the MFN and RMFD models, respectively
- __Detection of multiple faces:__ able to detect multiple faces in one frame
- __Support for detection in webcam stream:__ our app supports detection in images and video streams 
- __Support for detection of improper mask wearing:__ our MFN model is able to detect improper mask wearing including
  (1) uncovered chin, (2) uncovered nose, and (3) uncovered nose and mouth.

## About
This app detects human faces and proper mask wearing in images and webcam streams. 

Under the COVID-19 pandemic, wearing
mask has shown to be an effective means to control the spread of virus. The demand for an effective mask detection on 
embedded systems of limited computing capabilities has surged, especially in highly populated areas such as public 
transportations, hospitals, etc. Trained on MobileNetV2, a state-of-the-art lightweight deep learning model on 
image classification, the app is computationally efficient to deploy to help control the spread of the disease.

While many work on face mask detection has been developed since the start of the pandemic, few distinguishes whether a
mask is worn correctly or incorrectly. Given the discovery of the new coronavirus variant in UK, we aim to provide a 
more precise detection model to help strengthen enforcement of mask mandate around the world.

## Frameworks and Libraries
- __[OpenCV](https://opencv.org/):__ computer vision library used to process images
- __[OpenCV DNN Face Detector](https://github.com/opencv/opencv/blob/3.4.0/samples/dnn/resnet_ssd_face_python.py):__ 
  Caffe-based Single Shot-Multibox Detector (SSD) model used to detect faces
- __[Tensorflow](https://www.tensorflow.org/) / [Keras](https://keras.io/):__ deep learning framework used to build and train our models
- __[MobileNet V2](https://arxiv.org/abs/1801.04381):__ lightweight pre-trained model available in Keras Applications; 
  used as a base model for our transfer learning
- __[Dash](https://plotly.com/dash/):__ framework built upon Plotly.js, React and Flask; used built the demo app

## Datasets
We provide two models trained on two different datasets. 
Our RMFD dataset is built from the [Real World Masked Face Dataset](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset) 
and the MFN dataset is built from the [MaskedFace-Net](https://github.com/cabani/MaskedFace-Net) and 
[Flickr-Faces-HQ Dataset (FFHQ)](https://github.com/NVlabs/ffhq-dataset).

### RMFD dataset
This dataset consists of __4,408__ images:
- `face_no_mask`: 2,204 images
- `face_with_mask`: 2,204 images

Each image is a cropped real-world face image of unfixed sizes. The `face_no_mask` data is randomly sampled from the 90,568 no mask
data from the Real World Masked Face Dataset and the `face_with_mask` data entirely provided by the original dataset.

### MFN dataset
This dataset consists of __200,627__ images:
- `face_with_mask_correctly`: 67,193 images
- `face_with_mask_incorrectly`: 66,899 images
- `face_no_mask`: 66,535 images

The `face_with_mask_correctly` and `face_with_mask_incorrectly` classes consist of the resized 128*128 images from 
the original MaskedFace-Net work without any sampling. The `face_no_mask` is built from the 
Flickr-Faces-HQ Dataset (FFHQ) upon which the MaskedFace-Net data was created.
All images in MaskedFace-Net are morphed mask-wearing images and `face_with_mask_incorrectly` consists of 10% uncovered chin, 10% uncovered nose, and 80% uncovered nose and mouth images.

### Download
The dataset is now available [here](https://drive.google.com/file/d/1Y1Y67osv8UBKn_ANckCXPvY2aZqv1Cha/view?usp=sharing)! (June 11, 2021)

## Training Results
Both models are trained on 80% of their respectively dataset and validated/tested on the other 20%. They both achieved 99%
accuracy on their validation data.

MFN Model                             |  RMFD Model
:------------------------------------:|:--------------------------------------:
![](./figures/train_plot_MFN.jpg)   |  ![](./figures/train_plot_RMFD.jpg) 


However, the MFN model sometimes classifies `face_no_mask` as `face_with_mask_incorrectly`. Though this would not affect
goal of reminding people to wear mask properly, any suggestion to improve the model is welcomed.

## Requirements
This project is built using Python 3.8 on MacOS Big Sur 11.1. The training of the model is performed on custom GCP 
Compute Engine (8 vCPUs, 13.75 GB memory) with `tensorflow==2.4.0`. All dependencies and packages are listed in
`requirements.txt`. 

Note: We used `opencv-python-headless==4.5.1` due to an [issue](https://github.com/skvark/opencv-python/issues/423) 
with `cv2.imshow` on MacOS Big Sur. However, recent release of `opencv-python 4.5.1.48` seems to have fixed the problem.
Feel free to modify the `requirements.txt` before you install all the listed packages.

## Setup
1. Open your terminal, `cd` into where you'd like to clone this project, and clone the project:
```
$ git clone https://github.com/achen353/Face-Mask-Detector.git
```
2. Download and install Miniconda [here](https://docs.conda.io/en/latest/miniconda.html).
3. Create an environment with the packages on `requirements.txt` installed:
```
$ conda create --name env_name --file requirements.txt
```
4. Now you can `cd` into the clone repository to run or inspect the code.



### To detect masked faces in images
`cd` into `/src/` and enter the following command:
```
$ python detect_mask_images.py -i <image-path> [-m <model>] [-c <confidence>]
```

### To detect masked faces in webcam streams
`cd` into `/src/` and enter the following command:
```
$ python detect_mask_video.py [-m <model>] [-c <confidence>]
```

### To train the model again on the dataset
`cd` into `/src/` and enter the following command:
```
$ python train.py [-d <dataset>]
```
Make sure to modify the paths in `train.py` to avoid overwriting existing models.

Note: 
- `<image-path>` should be relative to the project root directory instead of `/src/`
- `<model>` should be of `str` type; accepted values are `MFN` and `RMFD` with default value `MFN`
- `<confidence>` should be `float`; accepting values between `0` and `1` with default value `0.5`
- `<dataset>` should be of `str` type; accepted values are `MFN` and `RMFD` with default value `MFN`

## Dash App Demo
The demo of the app is available [here](https://face-mask-detection-300106.wl.r.appspot.com); it is still under testing.

### Run the app yourself
1. Modify `app.run_server(host='0.0.0.0', port=8080, debug=True)` to `app.run_server(debug=True)`:
2. Run the app:
```
$ python main.py
```
3. Enter `http://127.0.0.1:8050/` in your browser to open the app on the Dash app's default host and port. Feel free to modify
the host and port number if the default port is taken.



### CODE EXPLANATION

from textwrap3 import dedent
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from tensorflow.keras.models import load_model
from src.detect_mask_image import detect_mask
import numpy as np
import base64
import cv2


app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server
app.config.suppress_callback_exceptions = True

global face_detector, mask_detector, vid_capture, original_opencv_img, annotated_img, confidence, status, alarm_on
prototxt_path = "./face_detector_model/deploy.prototxt"
weights_path = "./face_detector_model/res10_300x300_ssd_iter_140000.caffemodel"
face_detector = cv2.dnn.readNet(prototxt_path, weights_path)
mask_detector = load_model("./mask_detector_models/mask_detector_MFN.h5")
vid_capture = None
original_opencv_img = None
annotated_img = None
confidence = 50
status = 0
alarm_on = True


# Main App
app.layout = html.Div(
    children=[
        dcc.Interval(id="interval-update", interval=1000, n_intervals=0),
        html.Div(id="top-bar", className="row"),
        html.Div(
            className="container",
            children=[
                html.Div(
                    id="left-side-column",
                    className="eight columns",
                    children=[
                        html.Div(
                            id="header-section",
                            children=[
                                html.H4("TARP PROJECT "),
                                html.P(
                                    "Proper Mask Detection : To get started, select whether you want to detect an image or webcam feed, "
                                    "and choose the model and the confidence level of your face detector. "
                                    "The results will be marked with bounding boxes in realtime."
                                ),
                            ],
                        ),
                        html.Div(
                            id="video-mode",
                            children=[
                                html.Div(
                                    id="annotated-frame-container",
                                ),
                            ]
                        ),
                        html.Div(
                            id="image-mode",
                            children=[
                                html.Div(
                                    id="annotated-image-container",
                                ),
                                html.Div(
                                    id="upload-div",
                                    children=[dcc.Upload(
                                        id='upload-image',
                                        children=html.Div([
                                            'Drag and Drop or ',
                                            html.A('Select Image (.jpg/.jpeg/.png)')
                                        ]),
                                        style={
                                            'width': '100%',
                                            'height': '60px',
                                            'lineHeight': '60px',
                                            'borderWidth': '1px',
                                            'borderStyle': 'dashed',
                                            'borderRadius': '5px',
                                            'textAlign': 'center',
                                            'margin': '10px'
                                        },
                                        # Allow only one file to be uploaded
                                        multiple=False
                                    )],
                                    style={"visibility": "visible"}
                                ),
                            ],
                        ),
                        html.Div(id="hidden-div-1", style={"display": "none"}),
                        html.Div(id="hidden-div-2", style={"display": "none"}),
                        html.Div(id="hidden-div-3", style={"display": "none"}),
                    ],
                ),
                html.Div(
                    id="right-side-column",
                    className="four columns",
                    children=[
                        html.Div(
                            className="markdown-text",
                            children=[
                                dcc.Markdown(
                                    children=dedent(
                                        """
                                        ##### What is the app about?
            
                                        The app detects human faces and proper mask wearing in images and webcam 
                                        streams. Under the COVID-19 pandemic, the demand for an effective mask 
                                        detection on embedded systems of limited computing capabilities has surged.
                                        
                                        
                                        ##### Group members
            
                                        POOJA KUMARI (18BEI0117)
                                        RAMESH KUMAR (18BEE0027)
                                        R PRATUSHA PATNAIK (18BEE0001)
                                        
                                        Github Link for the [project repository](https://github.com/POOJAKUMARI1205/LIVE-FACE-MASK-DETECTOR).
                                        """
                                    )
                                )
                            ],
                        ),
                        html.Div(
                            className="control-element",
                            children=[
                                html.Div(children=["Detection Mode:"]),
                                dcc.RadioItems(
                                    id="detection-mode",
                                    options=[
                                        {
                                            "label": "Image",
                                            "value": "image",
                                        },
                                        {
                                            "label": "Webcam Video",
                                            "value": "video",
                                        },
                                    ],
                                    value="image",
                                ),
                            ],
                        ),
                        html.Div(
                            className="control-element",
                            children=[
                                html.Div(
                                    children=["Minimum Face Detector Confidence Threshold:"]
                                ),
                                html.Div(
                                    dcc.Slider(
                                        id="slider-face-detector-minimum-confidence-threshold",
                                        min=20,
                                        max=80,
                                        marks={
                                            i: f"{i}%"
                                            for i in range(20, 81, 10)
                                        },
                                        value=50,
                                        updatemode="drag",
                                    )
                                ),
                            ],
                        ),
                        html.Div(
                            className="control-element",
                            children=[
                                html.Div(children=["Mask Detector Model Selection:"]),
                                dcc.RadioItems(
                                    id="radio-item-mask-detector-selection",
                                    options=[
                                        {
                                            "label": "MFN Model",
                                            "value": "MFN",
                                        },
                                        {
                                            "label": "RMFD Model",
                                            "value": "RMFD",
                                        },
                                    ],
                                    value="MFN",
                                ),
                            ],
                        ),
                        html.Div(
                            id="alarm-control",
                            className="control-element",
                            children=[
                                html.Div(children=["Alarm on/off:"]),
                                dcc.RadioItems(
                                    id="radio-alarm-switch",
                                    options=[
                                        {
                                            "label": "Alarm on",
                                            "value": "on",
                                        },
                                        {
                                            "label": "Alarm off",
                                            "value": "off",
                                        },
                                    ],
                                    value="on",
                                ),
                            ],
                            style={
                                "visibility": "hidden",
                            },

                        ),
                        html.Div(
                            id="video-control",
                            children=[
                                html.Button(
                                    "Start Video",
                                    id="video-start-button",
                                    n_clicks=0,
                                    className="video-start-button",
                                ),
                                html.Div(
                                    id="hidden-gap",
                                    style={
                                        "margin-right": "1%",
                                        "margin-left": "1%",
                                        "visibility": "none",
                                    }
                                ),
                                html.Button(
                                    "Stop Video",
                                    id="video-stop-button",
                                    n_clicks=0,
                                    className="video-stop-button",
                                )
                            ],
                            style={
                                "visibility": "hidden",
                                "display": "flex",
                                "justify-content": "center",
                                "align-items": "center",
                                "margin-top": "3%",
                            },
                        ),
                        html.Div(
                            id="hidden-audio-div",
                            children=[
                                html.Audio(
                                    id="alert-audio",
                                    src=app.get_asset_url("no_mask_US_female.mp3"),
                                    autoPlay=False,
                                    controls=False,
                                    loop=True
                                )
                            ],
                            style={"visibility": "visible"}
                        )
                    ],
                ),
            ],
        ),
    ]
)


def detect_and_create_html_img(img, img_id):
    global status
    status, output_img = detect_mask(img, face_detector, mask_detector, confidence, False)
    output_base64 = cv2.imencode('.jpg', output_img)[1].tobytes()
    output_base64 = base64.b64encode(output_base64).decode('utf-8')
    output_url = "data:image/;base64,{}".format(output_base64)
    output_html_img = html.Img(
        id=img_id,
        className=img_id,
        src=output_url,
        style={
            "max-width": "100%",
            "height": "auto",
        }
    )
    return output_html_img


@app.callback(
    Output("upload-div", "style"),
    Input("detection-mode", "value"),
)
def display_upload(detection_mode):
    if detection_mode == "video":
        return {"visibility": "hidden"}
    return {"visibility": "visible"}


@app.callback(
    Output("video-control", "style"),
    Input("detection-mode", "value"),
)
def display_video_control(detection_mode):
    if detection_mode == "image":
        return {
            "visibility": "hidden",
            "display": "flex",
            "justify-content": "center",
            "align-items": "center",
            "margin-top": "3%",
        }
    return {
        "visibility": "visible",
        "display": "flex",
        "justify-content": "center",
        "align-items": "center",
        "margin-top": "3%",
    }


@app.callback(
    Output("alarm-control", "style"),
    Input("detection-mode", "value"),
)
def display_alarm_control(detection_mode):
    if detection_mode == "image":
        return {
            "visibility": "hidden",
        }
    return {
        "visibility": "visible",
    }


@app.callback(
    [
        Output("annotated-frame-container", "children"),
        Output("annotated-image-container", "children")
    ],
    [
        Input("upload-image", "contents"),
        Input("video-start-button", "n_clicks"),
        Input("video-stop-button", "n_clicks"),
        Input("interval-update", "n_intervals")
    ],
    [
        State("detection-mode", "value"),
        State("radio-alarm-switch", "value")
    ]
)
def update_image_or_frame(contents, start_n, end_n, n_intervals, detection_mode, alarm_state):
    global vid_capture, original_opencv_img, annotated_img, alarm_on
    ctx = dash.callback_context
    if ctx.triggered:
        input_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if input_id == "interval-update":
            if detection_mode == "video" and vid_capture is not None and vid_capture.isOpened():
                annotated_img = None
                flags, frame = vid_capture.read()
                annotated_frame = detect_and_create_html_img(frame, "annotated-frame")
                return annotated_frame, []
            elif detection_mode == "image" and annotated_img is not None:
                alarm_on = False
                return [], annotated_img
        if input_id == "video-start-button" and detection_mode == "video" \
                and (vid_capture is None or not vid_capture.isOpened()):
            annotated_img = None
            if alarm_state == "on":
                alarm_on = True
            else:
                alarm_on = False
            vid_capture = cv2.VideoCapture(0)
            return [], []
        if input_id == "video-stop-button" and detection_mode == "video" and \
                vid_capture is not None and vid_capture.isOpened():
            annotated_img = None
            alarm_on = False
            vid_capture.release()
            return [], []
        if input_id == "upload-image" and detection_mode == "image":
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            try:
                img_arr = np.frombuffer(decoded, dtype=np.uint8)
                img = cv2.imdecode(img_arr, flags=cv2.IMREAD_COLOR)
                original_opencv_img = img
                annotated_img = detect_and_create_html_img(img.copy(), "annotated-image")
                return [], annotated_img
            except:
                return [], [html.Div(["Please upload file of the following types: jpg, jpeg, png."])]
    return [], []


@app.callback(
    Output("hidden-div-1", "children"),
    Input("radio-item-mask-detector-selection", "value"),
)
def update_mask_detector(model):
    global mask_detector, annotated_img
    model_path = "./mask_detector_models/mask_detector_" + model + ".h5"
    mask_detector = load_model(model_path)
    if original_opencv_img is not None:
        annotated_img = detect_and_create_html_img(original_opencv_img.copy(), "annotated-image")
    return []


@app.callback(
    Output("hidden-div-2", "children"),
    Input("slider-face-detector-minimum-confidence-threshold", "value"),
)
def update_confidence(confidence_value):
    global confidence, annotated_img
    confidence = confidence_value * 0.01
    if original_opencv_img is not None:
        annotated_img = detect_and_create_html_img(original_opencv_img.copy(), "annotated-image")
    return []


@app.callback(
    Output("hidden-div-3", "children"),
    Input("radio-alarm-switch", "value"),
)
def update_alarm(alarm_state):
    global alarm_on
    if alarm_state == "on":
        alarm_on = True
    else:
        alarm_on = False

    return []


@app.callback(
    [
        Output("alert-audio", "src"),
        Output("alert-audio", "autoPlay")
    ],
    Input("interval-update", "n_intervals"),
    [
        State("detection-mode", "value"),
        State("radio-item-mask-detector-selection", "value")
    ]
)
def play_alert_audio(n_intervals, detection_mode, model):
    no_mask_src = app.get_asset_url("no_mask_US_female.mp3")
    mask_incorrect_src = app.get_asset_url("mask_incorrect_US_female.mp3")
    if detection_mode == "video":
        if model == "MFN":
            if status == 2 and alarm_on is True:
                return no_mask_src, True
            elif status == 1 and alarm_on is True:
                return mask_incorrect_src, True
        else:
            if status == 1 and alarm_on is True:
                return no_mask_src, True
    return "", False


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=True)

## Credits
- 口罩遮挡人脸数据集（[Real-World Masked Face Dataset](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset) ，RMFD）
- Adnane Cabani, Karim Hammoudi, Halim Benhabiles, and Mahmoud Melkemi, "MaskedFace-Net - A dataset of 
  correctly/incorrectly masked face images in the context of COVID-19", Smart Health, ISSN 2352-6483, 
  Elsevier, 2020, [DOI:10.1016/j.smhl.2020.100144](https://doi.org/10.1016/j.smhl.2020.100144)
- Karim Hammoudi, Adnane Cabani, Halim Benhabiles, and Mahmoud Melkemi,"Validating the correct wearing of protection 
  mask by taking a selfie: design of a mobile application "CheckYourMask" to limit the spread of COVID-19", 
  CMES-Computer Modeling in Engineering & Sciences, Vol.124, No.3, pp. 1049-1059, 
  2020, [DOI:10.32604/cmes.2020.011663](DOI:10.32604/cmes.2020.011663)
- [Flickr-Faces-HQ Dataset (FFHQ)](https://github.com/NVlabs/ffhq-dataset)
- [Face Mask Detection](https://github.com/chandrikadeb7/Face-Mask-Detection)
- [Object Detection](https://github.com/plotly/dash-sample-apps/tree/master/apps/dash-object-detection)

## License
[MIT © Andrew Chen](https://github.com/achen353/Face-Mask-Detector/blob/master/LICENSE)




