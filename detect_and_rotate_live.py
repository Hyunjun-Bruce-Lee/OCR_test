from ultralytics import YOLO
from copy import deepcopy
from pypylon import pylon # 3.0.1
import cv2 # 4.8.0
import imutils # 0.5.4
import moviepy.video.io.ImageSequenceClip # 1.0.3
import numpy as np # 1.19.5
import os
from datetime import datetime
import argparse

class rotate_img:
    def __init__(self):
        # Initialize feature detector and descriptor
        self.orb = cv2.ORB_create()
        # Initialize keypoint matcher
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def rotate_image(self, image, angle):
        # Get image dimensions
        height, width = image.shape[:2]
        # Calculate the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        # Apply the rotation to the image
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
        return rotated_image

    def __call__(self, base_img, target_img):
        # Save target_img for the result
        target_img_ori = deepcopy(target_img)
        # Convert image to gray scale
        base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        # Find keypoints and descriptors
        keypoints1, descriptors1 = self.orb.detectAndCompute(base_img, None)
        keypoints2, descriptors2 = self.orb.detectAndCompute(target_img, None)
        # Match keypoints
        matches = self.bf.match(descriptors1, descriptors2)
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        # Extract matched keypoints
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        # Estimate homography
        homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        # Extract rotation from homography matrix
        rotation_rad = -np.arctan2(homography[0, 1], homography[0, 0])
        rotation_deg = np.degrees(rotation_rad)
        # rotate image
        recovered_img = self.rotate_image(target_img_ori, rotation_deg)
        return recovered_img
    

class yolo_obb_detector:
    def __init__(self, weight_path):
        self.model = YOLO(weight_path)

    def __call__(self, img):
        results = self.model.predict(img)
        return results[0]

class basler:
    def __init__(self, img_size_multiplier = 1):
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice()) # declare camera instance
        self.converter = pylon.ImageFormatConverter() # declare image format converter
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned # add an arg to converter
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed # add an arg to converter
        
        self.height = img_size_multiplier * 480
        self.frame_holder = list()
        self.is_recording = False

        #################
        self.rotate_img = rotate_img()
        self.yolo = yolo_obb_detector("./runs/obb/train2/weights/best.pt")
        #################

        self.base_dir = f'C:/Users/{os.getlogin()}/Documents/basler_camera_data'
        if not os.path.isdir(self.base_dir):
            os.mkdir(self.base_dir)

        print(f"created data will be stored at {self.base_dir}")

    def check_dir_for_the_day(self):
        dir_for_the_day = self.base_dir + f'/{datetime.today().strftime("%Y%m%d")}'
        if not os.path.isdir(dir_for_the_day):
            os.mkdir(dir_for_the_day)
        return dir_for_the_day

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
            return '>>> camera is not running'
        
    def capture_current_frame(self, grab_result, dt_obj = None, display_img = False, save_captured_img = False):
        # capture and display(+save) camera's current frame\
        today_dir = self.check_dir_for_the_day()
        
        if dt_obj == None:
            dt_obj = datetime.now().replace(microsecond=0)

        file_name = f'{dt_obj.strftime(format = "%H%M%S")}.jpg'
        converted_result = self.converter.Convert(grab_result) # convert gained frame to cv2 readable format
        img_array = converted_result.GetArray() # get img array

        if display_img:
            # display image on a new window if param is true
            cv2.imshow(f"{file_name}", img_array)
            cv2.waitKey()
            cv2.destroyAllWindows()
        
        if save_captured_img:
            # save current frame as jpg file at C:/Users/User/Documents/basler_camera_capture_img
            if file_name not in os.listdir(today_dir):
                cv2.imwrite(f'{today_dir}/{file_name}',img_array)

        return img_array

    def __call__(self, auto_capture = False):
        print(">>> press q to stop \n >>> press c to capture and save current frame")
        
        init_time = datetime.now().replace(microsecond=0)
        
        if auto_capture != False:
            auto_capture, auto_capture_period = True, auto_capture

        if not self.get_cam_status():
            self.run_camera()
            print('cammera now running')

        rotated_img = np.zeros((480,480,3), dtype=np.uint8)

        while self.get_cam_status():
            grabResult = self.camera.RetrieveResult(100000, pylon.TimeoutHandling_ThrowException)
            
            if grabResult.GrabSucceeded():
                image = self.converter.Convert(grabResult)
                img = image.GetArray()
                img = imutils.resize(img, height=self.height)

                #### yolo detection
                result = self.yolo(img)
                if result.obb.cls.tolist():
                    try :
                        pred_label = result.names[int(result.obb.cls.tolist()[0])]
                        coords = result.obb.xyxyxyxy.reshape(4,2)
                        coords = [(round(int(x)),round(int(y))) for x,y in coords]

                        #### rotation
                        base_img = cv2.imread(f'./base_imgs/{pred_label}.jpg')
                        rotated_img = self.rotate_img(base_img, img)
                        
                        ### print label on right screen
                        cv2.putText(rotated_img, pred_label, (200, 240), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 5, cv2.LINE_AA)

                        ### display
                        for x,y in coords:
                            cv2.line(img, coords[0], coords[1], (0,0,255), thickness = 2)
                            cv2.line(img, coords[1], coords[2], (0,0,255), thickness = 2)
                            cv2.line(img, coords[2], coords[3], (0,0,255), thickness = 2)
                            cv2.line(img, coords[3], coords[0], (0,0,255), thickness = 2)
                    except:
                        print(result)

                numpy_horizontal = np.hstack((img, rotated_img))

                cv2.imshow(f'{self.camera.GetDeviceInfo().GetModelName()}', numpy_horizontal)
                key = cv2.waitKey(1)
                
                if key == 113: # press q to stop
                    end_time = datetime.now().replace(microsecond=0)
                    break
                elif (key == 118) and (not self.is_recording): # press v to start recording video
                    self.is_recording = True
                    record_init_time = datetime.now()
                    print('>>> strat recording')
                elif key == 115: # press s to stop recording video
                    self.is_recording = False
                    record_end_time = datetime.now()
                    print('>>> stop recording')
                
                if self.is_recording: # append current frame array to a holder
                    self.frame_holder.append(img)

                temp_dt = datetime.now().replace(microsecond=0)
                if (auto_capture) and (((temp_dt - init_time).seconds) % auto_capture_period == 0):
                    self.capture_current_frame(grabResult, dt_obj = temp_dt, display_img = False, save_captured_img = True)
                else:
                    if key == 99: # press c to capture and save
                        self.capture_current_frame(grabResult, display_img = False, save_captured_img = True)

            grabResult.Release()

        if self.frame_holder: # if holder has frame make video with those frames
            try :
                record_end_time
            except :
                record_end_time = datetime.now()

            video_runtime = round((record_end_time - record_init_time).total_seconds())
            video_len_frames = len(self.frame_holder)
            video_fps = round(video_len_frames/video_runtime)

            today_dir = self.check_dir_for_the_day()
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(self.frame_holder, fps=video_fps)
            clip.write_videofile(f'{today_dir}/{record_init_time.strftime("%H%M%S")}_{record_end_time.strftime("%H%M%S")}.mp4')

        cv2.destroyAllWindows()
        self.stop_camera()


# arguments 
parser = argparse.ArgumentParser(description='basler_camera')
parser.add_argument('--img_size_multiplier', default = 1, type = int, help='default 480 * n (n == 1)') ##### newally added
parser.add_argument('--autocapture_period', default = False, help='takes integer value. period by second')

args = parser.parse_args()


# main
if __name__ == '__main__':
    cam = basler(args.img_size_multiplier)
    if args.autocapture_period:
        assert args.autocapture_period.isdigit(), 'autocapture period must be integer'
        cam(auto_capture = int(args.autocapture_period))
    else:
        cam()

