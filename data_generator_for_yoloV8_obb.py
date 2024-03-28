import cv2
import numpy as np
import os
from copy import deepcopy
import argparse
from math import cos, sin, radians, floor
from datetime import datetime
import random


def resize_img(img, img_size): 
    # takes an array(img) as input
    return cv2.resize(img, dsize = img_size, interpolation = cv2.INTER_LANCZOS4)
    

def img2gray(img): 
    # converts img to gray scale
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

def increase_channel(img): 
    # convertes 2d gray img to 3d gray img
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def find_coord_candidates(img, lower_c = np.array([170,0,0]), upper_c = np.array([255,150,150])):
    # lower_c & uppder_c is BGR code for the color that you want to find
    # returns 2d array
    mask = cv2.inRange(img, lower_c, upper_c)
    coordinates = cv2.findNonZero(mask)
    number_of_corrds_detected, _, coord_shape = coordinates.shape
    coordinates = coordinates.reshape(number_of_corrds_detected, coord_shape)
    return coordinates


def coord_filter(coordinates, img_size): 
    # takes output from self.find_coord_candidates as an input
    # it devides img in to 4 areas(left top, right top, left bottom, right bottom)
    # respect to the areas devided above, it will calculate mean value of the coordinates for each area
    x_flag, y_flag = map(lambda x: round(x/2), img_size)
    filtered_coord = {'lt':list(), 'rt':list(), 'lb':list(), 'rb':list()}

    for x,y in coordinates:
        if (x<x_flag) and (y<y_flag): # left top
            filtered_coord['lt'].append([x,y])
        elif (x>x_flag) and (y<y_flag): # right top
            filtered_coord['rt'].append([x,y])
        elif (x>x_flag) and (y>y_flag): # right bottom
            filtered_coord['rb'].append([x,y])
        else: # left bottom
            filtered_coord['lb'].append([x,y])

    final_coords = dict()
    for position in ['lt','rt','lb','rb']:
        temp = filtered_coord[position]
        x_coords = [i[0] for i in temp]
        y_coords = [i[1] for i in temp]
        final_coords[position] = ((round(np.mean(x_coords)), round(np.mean(y_coords))))
    
    return final_coords


def overlay_dots(img, final_coords, color = (255,0,0), bigger_dot = False):
    # takes img and output from self.coord_filter as input
    img = deepcopy(img)
    for x,y in final_coords.values():
        if bigger_dot:
            for nx in range(x-2, x+2):
                for ny in range(y-2, y+2):
                    img[ny][nx][0],img[ny][nx][1],img[ny][nx][2] = color
        else:
            img[y][x][0],img[y][x][1],img[y][x][2] = color
    return img


def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


def rotate_coord(x, y, xc, yc, angle):
    # x,y is the coordnate that you want to rotate
    # xc,yc is the centroid coordinate for the background
    # it iwll rotate anticlockwise (if you want to rotate clockwist change (angle) to (360-angle))
    new_x = (x - xc) * cos(radians(360 - angle)) - (y - yc) * sin(radians(360 - angle)) + xc
    new_y = (x - xc) * sin(radians(360 - angle)) + (y - yc) * cos(radians(360 - angle)) + yc
    return int(new_x), int(new_y)


def cal_coord(cord_info, angle, cx, cy): 
    # c for centroid value for background. if img is not perfect square need to change the code
    temp_dict = dict()
    for position in ['lt','rt','lb','rb']:
        x,y = cord_info[position]
        x,y = rotate_coord(x,y,cx,cy,angle)
        temp_dict[position] = (x,y)
    return temp_dict


def create_coord_string(cord_info, label_list, label):
    label_idx = label_list.index(label)
    x1,y1 = cord_info['lt']
    x2,y2 = cord_info['rt']
    x3,y3 = cord_info['lb']
    x4,y4 = cord_info['rb']
    return f"{label_idx} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}"


def normalize_coord(coord, img_size):
    x = (img_size[0] - coord[0])/img_size[0]
    y = (img_size[1] - coord[1])/img_size[1]
    return x,y


def random_sampling(len_idx, data_ratio = (0.8,0.1,0.1)):
    train, valid, test = list(), list(), list()
    temp_list = range(len_idx)
    remainings = random.sample(temp_list, len(temp_list) - floor(len(temp_list)*data_ratio[0]))
    test_idxs = random.sample(remainings, len(remainings) - floor(len(temp_list)*0.1))
    for idx in temp_list:
        if idx in test_idxs:
            test.append(idx)
        elif idx in remainings:
            valid.append(idx)
        else:
            train.append(idx)
    return train,valid,test


class data_generator:
    def __init__(self, base_dir, img_size = (480,480)): 
        self.base_dir = base_dir
        self.label_nms = [i.split('.')[0] for i in os.listdir(base_dir) if i.endswith('.jpg')]
        self.resized_imgs = list()
        self.skip_data = list()
        self.img_size = img_size
        self.yolo_data_dir = self.make_yolo_data_dir()
        self.cread_yaml(self.yolo_data_dir, self.label_nms)
    
    def make_yolo_data_dir(self):
        time = datetime.now().strftime("%Y%m%d%H%M%S")
        save_dir = f"{self.base_dir}/yolo_data_{time}"
        os.mkdir(save_dir)
        os.mkdir(save_dir + '/train')
        os.mkdir(save_dir + '/train' + '/images')
        os.mkdir(save_dir + '/train' + '/labels')
        os.mkdir(save_dir + '/valid')
        os.mkdir(save_dir + '/valid' + '/images')
        os.mkdir(save_dir + '/valid' + '/labels')
        os.mkdir(save_dir + '/test')
        os.mkdir(save_dir + '/test' + '/images')
        os.mkdir(save_dir + '/test' + '/labels')
        return save_dir

    def cread_yaml(self, save_dir, label_list):
        with open(f'{save_dir}/data.yaml', 'w+') as f:
            f.write("train: ./train/images" + '\n')
            f.write("valid: ./valid/images" + '\n')
            f.write("test: ./test/images" + '\n')
            f.write('\n')
            f.write(f'nc: {len(self.label_nms)}' + '\n')
            f.write(f'names: {label_list}')

    def __call__(self, data_ratio = (0.8,0.1,0.1), angles = range(0,360,10)):
        for nm in self.label_nms:
            target_img = cv2.imread(f'{self.base_dir}/{nm}.jpg')
            resized_img = resize_img(target_img, self.img_size)
            self.resized_imgs.append(resized_img)
            gray_img = img2gray(resized_img)
            gray_img_3d = increase_channel(gray_img) # not nessesary as gray images will be imported again after (it will be imported as 3D(BGR) array)
            
            if not os.path.isdir(f"{self.yolo_data_dir}/Gray_imgs"):
                os.mkdir(f"{self.yolo_data_dir}/Gray_imgs")
            cv2.imwrite(f"{self.yolo_data_dir}/Gray_imgs/{nm}.jpg", gray_img_3d)
        
        k = input(f"Gray images are created at {self.yolo_data_dir}/Gray_imgs. \n Draw 4 dots(vivid blue) surrounding the object \n press any key when the drawing is done (if you want to keep the dotted image after the process type in 'y')")

        test_label, valid_label, train_label = list(), list(), list()
        test_img, valid_img, train_img = list(), list(), list()
        for idx, nm in enumerate(self.label_nms):
            target_img_gray = cv2.imread(f'{self.yolo_data_dir}/Gray_imgs/{nm}.jpg')
            coordinates = find_coord_candidates(target_img_gray)
            ordinary_coord = coord_filter(coordinates, self.img_size)

            ord_dot = overlay_dots(target_img_gray, ordinary_coord, (0,0,255), True)
            if not os.path.isdir(f'{self.yolo_data_dir}/gray_with_dots'):
                os.mkdir(f'{self.yolo_data_dir}/gray_with_dots')
            cv2.imwrite(f'{self.yolo_data_dir}/gray_with_dots/{nm}.jpg',ord_dot)

            temp_labels, temp_imgs = list(), list()
            for angle in angles:
                roteated_coord = cal_coord(ordinary_coord, angle, round(self.img_size[0]/2), round(self.img_size[1]/2))
                rotated_img = rotate_image(self.resized_imgs[idx], angle)
                if k == 'y':
                    rotated_dot = overlay_dots(rotated_img, roteated_coord, (0,0,255), True)
                    if not os.path.isdir(f'{self.yolo_data_dir}/img_with_dots'):
                        os.mkdir(f'{self.yolo_data_dir}/img_with_dots')
                    cv2.imwrite(f'{self.yolo_data_dir}/img_with_dots/{nm}_a{angle}.jpg',rotated_dot)
                
                for key in roteated_coord.keys():
                    roteated_coord[key] = normalize_coord(roteated_coord[key], self.img_size)
                
                temp_labels.append(create_coord_string(roteated_coord, self.label_nms, nm))
                temp_imgs.append(rotated_img)

            train_idx, valid_idx, _ = random_sampling(len(temp_imgs), data_ratio)
            for idx in range(len(temp_imgs)):
                if idx in train_idx:
                    train_label.append(temp_labels[idx])
                    train_img.append(temp_imgs[idx])
                elif idx in valid_idx:
                    valid_label.append(temp_labels[idx])
                    valid_img.append(temp_imgs[idx])
                else:
                    test_label.append(temp_labels[idx])
                    test_img.append(temp_imgs[idx])

        for idx in range(len(train_img)):
                with open(self.yolo_data_dir + '/train' + '/labels' + f'/img_{idx}.txt', 'w+') as f:
                    f.write(train_label[idx]) 
                cv2.imwrite(self.yolo_data_dir + '/train' + '/images' + f'/img_{idx}.jpg', train_img[idx])
        
        for idx in range(len(valid_img)):
                with open(self.yolo_data_dir + '/valid' + '/labels' + f'/img_{idx}.txt', 'w+') as f:
                    f.write(valid_label[idx]) 
                cv2.imwrite(self.yolo_data_dir + '/valid' + '/images' + f'/img_{idx}.jpg', valid_img[idx])

        for idx in range(len(test_img)):
                with open(self.yolo_data_dir + '/test' + '/labels' + f'/img_{idx}.txt', 'w+') as f:
                    f.write(test_label[idx]) 
                cv2.imwrite(self.yolo_data_dir + '/test' + '/images' + f'/img_{idx}.jpg', test_img[idx])



# arguments 
parser = argparse.ArgumentParser(description='data generator for yolo-obb object detection')
parser.add_argument('--base_dir', default = './data', help='-')

args = parser.parse_args()


# main
if __name__ == '__main__':
    generator = data_generator(args.base_dir)
    generator()