import deeplabcut
f = open('cheeseboard/files.txt', 'r')
lines = [x.strip() for x in f.readlines()]
config_path = '/home/prez/MouseTrack-Prez-2018-12-07/config.yaml'
deeplabcut.analyze_videos(config_path, lines)
