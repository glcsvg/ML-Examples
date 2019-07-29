from moviepy.editor import VideoFileClip, concatenate_videoclips

file_name = r"C:\Users\Doruk\Desktop\Logitech Webcam\a kapı video1.mp4"
j=0 

for i in range(1,5):    
    name = 'clip' + str(i)
    name = VideoFileClip(file_name).subclip(j,j+5)
    print(j)
    print(name)
    j = j + 3
    
    
# 5 10 13 18