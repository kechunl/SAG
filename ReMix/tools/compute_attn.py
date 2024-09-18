import os
import json
import numpy as np
import xml.etree.ElementTree as ET
import openslide
import cv2
import argparse
import glob
import pdb
from tqdm import tqdm 
from PIL import Image
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu

def camelyon16xml2json(inxml):
    """
    Convert an annotation of camelyon16 xml format into a json format.
    Arguments:
        inxml: string, path to the input camelyon16 xml format
    """
    root = ET.parse(inxml).getroot()
    annotations_tumor = \
        root.findall('./Annotations/Annotation[@PartOfGroup="Tumor"]')
    annotations_0 = \
        root.findall('./Annotations/Annotation[@PartOfGroup="_0"]')
    annotations_1 = \
        root.findall('./Annotations/Annotation[@PartOfGroup="_1"]')
    annotations_2 = \
        root.findall('./Annotations/Annotation[@PartOfGroup="_2"]')
    annotations_positive = \
        annotations_tumor + annotations_0 + annotations_1
    annotations_negative = annotations_2

    json_dict = {}
    json_dict['positive'] = []
    json_dict['negative'] = []

    for annotation in annotations_positive:
        X = list(map(lambda x: float(x.get('X')),
                    annotation.findall('./Coordinates/Coordinate')))
        Y = list(map(lambda x: float(x.get('Y')),
                    annotation.findall('./Coordinates/Coordinate')))
        vertices = np.round([X, Y]).astype(int).transpose().tolist()
        name = annotation.attrib['Name']
        json_dict['positive'].append({'name': name, 'vertices': vertices})

    for annotation in annotations_negative:
        X = list(map(lambda x: float(x.get('X')),
                    annotation.findall('./Coordinates/Coordinate')))
        Y = list(map(lambda x: float(x.get('Y')),
                    annotation.findall('./Coordinates/Coordinate')))
        vertices = np.round([X, Y]).astype(int).transpose().tolist()
        name = annotation.attrib['Name']
        json_dict['negative'].append({'name': name, 'vertices': vertices})

    return json_dict


def xml2mask(wsi_path, xml_path, level=4):
    dicts = camelyon16xml2json(xml_path)

    slide = openslide.OpenSlide(wsi_path)
    w, h = slide.level_dimensions[level]
    mask_tumor = np.zeros((h, w)) # the init mask, and all the value is 0
    
    factor = slide.level_downsamples[level]# get the factor of level * e.g. level 6 is 2^6

    tumor_polygons = dicts['positive']
    for tumor_polygon in tumor_polygons:
        # plot a polygon
        vertices = np.array(tumor_polygon["vertices"]) / factor
        vertices = vertices.astype(np.int32)
        cv2.fillPoly(mask_tumor, [vertices], (255))
    
    mask_tumor = mask_tumor[:] > 127
    return mask_tumor


def get_tissue_mask(wsi_path, level=4, RGB_min=50):
    slide = openslide.OpenSlide(wsi_path)

    # note the shape of img_RGB is the transpose of slide.level_dimensions
    img_RGB = np.array(slide.read_region((0, 0),
                           level,
                           slide.level_dimensions[level]).convert('RGB'))

    img_HSV = rgb2hsv(img_RGB)

    background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
    background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
    background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
    tissue_RGB = np.logical_not(background_R & background_G & background_B)
    tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
    min_R = img_RGB[:, :, 0] > RGB_min
    min_G = img_RGB[:, :, 1] > RGB_min
    min_B = img_RGB[:, :, 2] > RGB_min

    tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B

    return tissue_mask


def get_attn_map(mask, tile_size, basename, output_path, datapath):
    patch_list = sorted(glob.glob(os.path.join(datapath, basename, '*.jpeg')))
    attn_score = []
    for patch_path in patch_list:
        patch_name = os.path.basename(patch_path)
        col, row = patch_name.split('.')[0].split('_')
        col, row = int(col), int(row)
        crop = mask[row*tile_size:(row+1)*tile_size, col*tile_size:(col+1)*tile_size]
        attn_score.append(np.sum(crop)/(tile_size**2))
    attn_score = np.array(attn_score)
    attn_score /= np.sum(attn_score)
    np.save(output_path, attn_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='/projects/melanoma/Kechun/Camelyon/MIL/WSI/Camelyon16/single', help='dataset path')
    parser.add_argument('--xml_dir', type=str, default='/projects/digipath2/CAMELYON/CAMELYON16/annotations', help='metastasis xml')
    parser.add_argument('--output_dir', type=str, default='/projects/melanoma/Kechun/Camelyon/MIL/Attn/metastasis', help='output folder')
    parser.add_argument('--tile_size', type=int, default=224, help='tile size')
    parser.add_argument('--level', type=int, default=4, help='magnification level to process on')
    parser.add_argument('--method', default='cancer', choices=['cancer', 'tissue_mask'])
    parser.add_argument('--wsi_dir', type=str, default='/projects/digipath2/CAMELYON/CAMELYON16/images')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    image_list = sorted(glob.glob(os.path.join(args.datapath, 'test_*'))+glob.glob(os.path.join(args.datapath, 'tumor_*'))+glob.glob(os.path.join(args.datapath, 'normal_*')))
    for image_path in tqdm(image_list, total=len(image_list)):
        basename = os.path.basename(image_path)
        output_path = os.path.join(args.output_dir, basename+'.npy')
        if os.path.exists(output_path):
            continue
        wsi_path = os.path.join(args.wsi_dir, basename+'.tif')
        xml_path = os.path.join(args.xml_dir, basename+'.xml')
        
        if args.method == 'tissue_mask':
            mask = get_tissue_mask(wsi_path, level=args.level)
        else:
            if not os.path.exists(xml_path):
                continue
            mask = xml2mask(wsi_path, xml_path, level=args.level) # level=4 --> 2.5x, level=1 --> 20x
        # mask: 0 or 1
        
        tile_size = args.tile_size // (2**(args.level-1))

        get_attn_map(mask, tile_size, basename, output_path, args.datapath)
