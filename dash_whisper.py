#!/usr/bin/env python3
# coding: utf-8

import base64
import os

import dash_bootstrap_components as dbc
import whisper
from dash import Dash, Input, Output, State, dcc, html
from flask import Flask
from moviepy.editor import AudioFileClip, VideoFileClip

version = "alpha.0.0"
self_name = os.path.basename(__file__)[:-3]
path = os.path.dirname(os.path.abspath(__file__))
folder = f"{path}/dash_whisper/"
if not os.path.exists(folder):
    os.makedirs(folder)

server = Flask(__name__)
app = Dash(
    __name__,
    server=server,
    title="Dash Whisper",
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.COSMO],
    meta_tags=[
        {"charset": "utf-8"},
        {"name": "viewport", "content": "width=device-width, initial-scale=1, shrink-to-fit=no"},
    ],
)
app.layout = dbc.Container(
    children=[
        dcc.Store(id="out", data=[]),
        dbc.Row(html.H1(html.I("Dash Whisper")), style={"text-align":"center"}),
        dbc.Row(
            html.Div(
                html.Div(id="output"),
                style={"height":"100%", "overflowY":"auto", "display":"flex"}),
            style={"width":"100%", "height":"calc(100vh - 200px)", "margin-top":"25px", "margin-bottom":"25px"}),
        dbc.Row(
            dcc.Upload(
                id="upload",
                children=dbc.Button("Audio or Video Files"),
            ),
            style={"text-align":"center"},
        )
    ]
)

@app.callback(
    [
        Output("upload", "filename"),
        Output("upload", "contents"),
        Output("output", "children"),
    ],
    [
        Input("upload", "filename"),
    ],
    [
        State("upload", "contents"),
    ]
)
def transcript(file, content):
    out = ["", "", ""]
    if file and content:
        file_name = file.split(".")[0]
        media_source = folder + os.sep + file
        audio_mp3 = folder + os.sep + file_name + ".mp3"
        cont = base64.b64decode(content.split(",")[-1])
        if "audio" in content:
            open(media_source, "wb").write(cont)
            audio = AudioFileClip(media_source)
            audio.write_audiofile(audio_mp3, codec="mp3")
            model = whisper.load_model("medium")
            result = model.transcribe(audio_mp3, fp16=False)
            os.remove(media_source)
            os.remove(audio_mp3)
            out = ["", "", result["text"]]
        elif "video" in content:
            open(media_source, "wb").write(cont)
            video = VideoFileClip(media_source)
            audio = video.audio
            audio.write_audiofile(audio_mp3, codec="mp3")
            model = whisper.load_model("medium")
            result = model.transcribe(audio_mp3, fp16=False)
            os.remove(media_source)
            os.remove(audio_mp3)
            out = ["", "", result["text"]]
        else:
            out = ["", "", "This file is not an audio or a video"]
    return out


if __name__ == "__main__":
    if len(os.sys.argv) == 1:
        app.run(host="127.0.0.1", port="8888", debug=True)
    elif len(os.sys.argv) == 2:
        host = os.sys.argv[1]
        os.system(f"gunicorn {self_name}:server -b {host}:8888 --reload --timeout 120")
    elif len(os.sys.argv) == 3:
        host = os.sys.argv[1]
        port = int(os.sys.argv[2])
        os.system(f"gunicorn {self_name}:server -b {host}:{port} --reload --timeout 120")
