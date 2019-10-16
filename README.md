![GitHub issues](https://img.shields.io/github/issues/mrsupiri/r2d2)
[![codebeat badge](https://codebeat.co/badges/678e76b0-303f-4f75-ab50-31759da05ed8)](https://codebeat.co/projects/github-com-mrsupiri-r2d2-master)
[![Requirements Status](https://requires.io/github/mrsupiri/R2D2/requirements.svg?branch=master)](https://requires.io/github/mrsupiri/R2D2/requirements/?branch=master)
![GitHub license](https://img.shields.io/github/license/mrsupiri/R2D2)
![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi-brightgreen)
[![Say Thanks!](https://img.shields.io/badge/Say%20Thanks-!-1EAEDB.svg)](https://saythanks.io/to/mrsupiri)
[![Discord Shield](https://discordapp.com/api/guilds/589829086583455757/widget.png?style=shield)](https://discord.gg/8dQCZzk)
[![Twitter Follow](https://img.shields.io/twitter/follow/mrsupiri?style=social)](https://twitter.com/mrsupiri)


# R2D2
Line Following Robot Powered By OpenCV and Machine Learing. This was Originally Developed for [RoboFest 2018](http://www.robofest.lk/) Organized by SLIIT 

<img src="https://cdn.iconicto.com/GitHub/R2D2/20170831_195059.jpg" width="200">

## Installation

Use the python package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.

```bash
pip install -r requirements.txt
```

## Technologies Used
- Raspberry Pi
- Numpy
- OpenCV
- Pandas
- Arduino
- Sklearn
- Socket IO

## Usage
- Adjust pins number in R2D2.py to Match your pin map
- Change IP in GrabFrame.py, lineFollower.py, Collectdata.py  to your Computers IP
- Run GrabFrame.py on PC
- Run collectData.py or lineFollower.py (depend on what you want to do, This is pre Trained to Run on simple Loop)
- If you Want to Collect Data you Should change collectData Variable to True on both collectData.py and lineFollower.py 

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
