# created by Sijmen van der Willik
# 06/08/2018 14:51

from moviepy.editor import VideoFileClip
from IPython.display import HTML

import predict_vehicle

clipped_vid = VideoFileClip("project_video.mp4")

annotated_clip = clipped_vid.fl_image(predict_vehicle.predict_vehicle)  # NOTE: this function expects color images!!
annotated_clip.write_videofile("clip_output_vid.mp4", audio=False)

