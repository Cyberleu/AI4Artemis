import os
os.environ['DISPLAY'] = ':0'
os.environ['XAUTHORITY']='/run/user/1000/gdm/Xauthority'
import pyautogui
import time
while(1):
    time.sleep(15)
    pyautogui.click()