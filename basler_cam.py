from pypylon import pylon # 3.0.1
import cv2 # 4.8.0
import imutils # 0.5.4
import os
from datetime import datetime
import argparse

class basler:
    def __init__(self):
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice()) # declare camera instance
        self.converter = pylon.ImageFormatConverter() # declare image format converter
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned # add an arg to converter
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed # add an arg to converter

    def get_cam_status(self):
        # check whether if camera is running or not
        return self.camera.IsGrabbing() 

    def run_camera(self):
        # activate camera
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        return '>>> starting camera'
    
    def stop_camera(self):
        # de-activate camera
        if self.camera.IsGrabbing():
            self.camera.StopGrabbing()
            return '>>> camera stopped'
        else:
            return ">>> camera is not running"
        
    def capture_current_frame(self, display_img = False, save_captured_img = False):
        # capture and display(+save) camera's current frame
        grab_result = self.camera.RetrieveResult(500000, pylon.TimeoutHandling_ThrowException) # get current frame
        converted_result = self.converter.Convert(grab_result) # convert gained frame to cv2 readable format
        img_array = converted_result.GetArray() # get img array
        dt_obj = datetime.now()

        if display_img :
            # display image on a new window if param is true
            cv2.imshow(f"{dt_obj.hour}-{dt_obj.minute}-{dt_obj.second}", img_array)
            cv2.waitKey()
            cv2.destroyAllWindows()
        
        if save_captured_img:
            # save current frame as jpg file at ./capture_img
            if not os.path.isdir("./capture_img"):
                os.mkdir("./capture_img")                 
            cv2.imwrite(f'./capture_img/{dt_obj.hour}-{dt_obj.minute}-{dt_obj.second}.jpg',img_array)

        return img_array

    def __call__(self, auto_capture = False):
        print(""">>> press q to stop \n 
                 >>> press c to capture and save current frame""")
        
        init_time = datetime.now()
        
        if str(auto_capture).isdigit():
            auto_capture, auto_capture_period = True, auto_capture

        
        if not self.get_cam_status():
            self.run_camera()
            print("cammera now running")
        
        while self.get_cam_status():
            grabResult = self.camera.RetrieveResult(500000, pylon.TimeoutHandling_ThrowException)
            
            if grabResult.GrabSucceeded():
                image = self.converter.Convert(grabResult)
                img = image.GetArray()
                img = imutils.resize(img, height=480)
                cv2.imshow('hello', img)
                k = cv2.waitKey(1)
                
                if k == 113: # press q to stop
                    break

                if (auto_capture) and (((datetime.now() - init_time).seconds) % auto_capture_period == 0): # time condition need to be added
                    self.capture_current_frame(display_img = False, save_captured_img = True)
                else:
                    if k == 99: # press c to capture and save
                        self.capture_current_frame(display_img = False, save_captured_img = True)
            
            grabResult.Release()
        cv2.destroyAllWindows()
        self.stop_camera()


# arguments 
parser = argparse.ArgumentParser(description='basler_camera')
parser.add_argument('--autocapture_period', default = False, help='by second')
args = parser.parse_args()


# main
if __name__ == '__main__':
    cam = basler()
    if args.autocapture_period:
        cam(auto_capture = int(args.autocapture_period))
    else:
        cam()