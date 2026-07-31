from utils import read_video, save_video
from tracking import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np
import pandas as pd
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import speed_calculator

def main():
    VIDEO_NAME = "bundesliga"

    video_frames = read_video(f'/Users/aakash/football_larp/input_videos/{VIDEO_NAME}.mp4')
    print("Video read, frame count:", len(video_frames))

    tracker = Tracker('/Users/aakash/football_larp/models/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path=f'/Users/aakash/football_larp/stubs/{VIDEO_NAME}_tracks.pkl')
    print("Tracks obtained")

    tracker.add_position_to_tracks(tracks)

    #camera movement
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path=f'stubs/{VIDEO_NAME}_camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)

    #perspective transform
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    #interpolate ball positions
    tracks['ball'] = tracker.ball_interpolate(tracks['ball'])

    #speed estimator
    speed_estimator = speed_calculator()
    speed_estimator.add_speed_to_tracks(tracks)

    #team assigner

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    #assign ball to player, basically ball possesion for a single player
    player_assigner = PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):

        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)


        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)

    #draw output videos
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    #draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    #draw speed
    speed_estimator.draw_speed(output_video_frames, tracks)

    save_video(output_video_frames, f'/Users/aakash/football_larp/output_videos/{VIDEO_NAME}_output_video.mp4')
    print("Video saved")

if __name__ == '__main__':
    main()