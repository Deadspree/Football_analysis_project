from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2
import numpy as np
import pandas as pd
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width
class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def interpolate_ball_position(self, ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[])for x in ball_positions]
        #Get track_id 1, if there is no track id, is going to be empty dict
        #in dictionary if there is no bbox, get a empty list
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1', 'x2', 'y2'])

        #Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {'bbox':x}}for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        batch_size = 20 #Detect(20x20 frames at a time to avoide memeory problem)
        detections = []
        #Make detection batch by batch
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
            

        return detections

    def get_object_tracks(self, frames, read_from_stub =False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players":[], #the output is gonna look like this:     (0:{"bbox":[0,0,0,0]}, 1:{"bbox":[0,0,0,0]}) dictionary of 1 frame, frame 2 ,3 looks the same and it all stored in the tracks
            "referees":[],
            "ball":[]
        }
        #Overwrite Goalkeeper as a player
        for frame_num, detection in enumerate(detections): #Enumerate put in the index of the list
            cls_names = detection.names #(0:ball, 1: player,..)
            cls_names_inv = {v:k for k, v in cls_names.items()} #(ball:0, player:1, ...)

            #Covnert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            #Convert goalkeeper to palyer object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] =='goalkeeper':
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]#Change class_id of goalkeeper to clas_id of player

            #Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            #Each object has different tracking ID(1,2,3,4) even if they are in the same class

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0]
                cls_id = frame_detection[3]
                if cls_id == cls_names_inv['ball']:
                    tracks['ball'][frame_num][1] = {"bbox":bbox}
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks, f)       #We gonna save it right here     
        return tracks
 
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])

        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(frame, center = (x_center, y2), axes = (int(width), int(0.35*width)), angle = 0.0, startAngle= -45, endAngle= 235, color= color, thickness=2, lineType=cv2.LINE_4)

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) + 15
        y2_rect = (y2+ rectangle_height//2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                 color,
                 cv2.FILLED)
             
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                (0,0,0),
                2


            )


        return frame

    def draw_triangle(slef, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10, y-20],
            [x+10, y-20]
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2) # Only draw the edge

        return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy() #To not pollute the video frame that are coming in
            #Draw a circle below the user

            player_dict = tracks['players'][frame_num]
            ball_dict = tracks['ball'][frame_num]
            referee_dict = tracks["referees"][frame_num]

            #Draw players ellpise
            for track_id, player in player_dict.items():
                color = player.get('team_color', (0,0,255))
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id )
            #Draw referees ellipse
            for _ , referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0,255,255) )
            #Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0,255,0))
            output_video_frames.append(frame)
        
        return output_video_frames




