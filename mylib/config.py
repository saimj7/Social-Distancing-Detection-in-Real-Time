# base path to YOLO directory
MODEL_PATH = "yolo"
# initialize minimum probability to filter weak detections along with
# the threshold when applying non-maxima suppression
MIN_CONF = 0.3
NMS_THRESH = 0.3

#===============================================================================
#=================================\CONFIG./=====================================
""" Below are your desired config. options to set for real-time inference """
# To count the total number of people (True/False).
People_Counter = True
# Threading ON/OFF. Please refer 'mylib>thread.py'.
Thread = False
# Set the threshold value for total violations limit.
Threshold = 15
# Enter the ip camera url (e.g., url = 'http://191.138.0.100:8040/video');
# Set url = 0 for webcam.
url = ''
# Turn ON/OFF the email alert feature.
ALERT = False
# Set mail to receive the real-time alerts. E.g., 'xxx@gmail.com'.
MAIL = ''
# Set if GPU should be used for computations; Otherwise uses the CPU by default.
USE_GPU = True
# Define the max/min safe distance limits (in pixels) between 2 people.
MAX_DISTANCE = 80
MIN_DISTANCE = 50
#===============================================================================
#===============================================================================
