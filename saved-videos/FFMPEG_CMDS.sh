ffmpeg -i mean_cbf_collides_bayes_cbf_is_safe.mp4 -filter_complex "drawtext=text=Mean CBF:x=.2*w:y=.05*h:fontsize=32, drawtext=text=Bayes CBF:x=0.7*w:y=0.05*h:fontsize=32" tmp.mp4
# Combine two videos
ffmpeg -i learning_helps_avoid_getting_stuck_no_learning.mp4 -i learning_helps_avoid_getting_stuck.mp4 -t 00:00:08 -filter_complex hstack=inputs=2 learning_helps_avoid_getting_stuck_cmp.mp4

# Label combined video
fmpeg -i learning_helps_avoid_getting_stuck_cmp.mp4 -filter_complex "drawtext=text=No Learning:x=.2*w:y=.05*h:fontsize=32, drawtext=text=With Learning:x=0.7*w:y=0.05*h:fontsize=32" tmp.mp4
