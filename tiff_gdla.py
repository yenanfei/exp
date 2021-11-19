# coding=utf-8
import os
from copy import deepcopy

import cv2
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly
import numpy as np
from tqdm import tqdm


def get_jingwei_by_coord(geotransform, x, y):
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    rx = geotransform[2]
    ry = geotransform[4]
    xgeo = originX + x * pixelWidth + y * rx
    ygeo = originY + x * ry + y * pixelHeight
    return xgeo, ygeo


def get_value_by_jingwei(geotransform, xgeo, ygeo):
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    rx = geotransform[2]
    ry = geotransform[4]
    if rx == ry == 0:
        x = (xgeo - originX) / pixelWidth
        y = (ygeo - originY) / pixelHeight
        return x, y
    else:
        raise Exception


def writeTiff(im_data, im_width, im_height, im_bands, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


def readImage(dom_path, dsm_path, out_dir):
    dom = gdal.Open(dom_path, GA_ReadOnly)
    dsm = gdal.Open(dsm_path, GA_ReadOnly)
    if dom is None or dsm is None:
        raise Exception
    else:
        domgeoTransform = dom.GetGeoTransform()
        dsmgeoTransform = dsm.GetGeoTransform()
        # im_proj = dom.GetProjection()  # 获取投影信息
        # print(im_proj)
        # print("Image height:" + dataset.RasterYSize.__str__() + " Image width:" + dataset.RasterXSize.__str__())
        # 获取影像的第i+1个波段
        band_dom = dom.GetRasterBand(1)
        band_dsm = dsm.GetRasterBand(1)
        # 读取第i+1个波段数据
        band_dom_data = band_dom.ReadAsArray(0, 0, band_dom.XSize, band_dom.YSize)
        band_dsm_data = band_dsm.ReadAsArray(0, 0, band_dsm.XSize, band_dsm.YSize)
        ori_dsm_data = deepcopy(band_dom_data)
        band_dom_data = (band_dom_data * 0).astype(np.float32)
        for i in tqdm(range(band_dom_data.shape[0])):
            for j in range(band_dom_data.shape[1]):
                # if band_dom_data[i][j] == 0:
                #     continue
                x_geo, y_geo = get_jingwei_by_coord(domgeoTransform, j, i)
                y, x = get_value_by_jingwei(dsmgeoTransform, x_geo, y_geo)
                if x > 0 and y > 0 and round(x) < band_dsm_data.shape[0] and round(y) < band_dsm_data.shape[1]:
                    h = band_dsm_data[round(x)][round(y)]
                    if h < -9999:
                        continue
                    else:
                        band_dom_data[i][j] = h
        for i in tqdm(range(0, band_dom_data.shape[0], 10)):
            if i + 768 < band_dom_data.shape[0]:
                continue
            for j in range(0, band_dom_data.shape[1], 10):
                if j + 768 < band_dom_data.shape[1]:
                    continue
                dsm_array = ori_dsm_data[i:i + 768, j:j + 768]
                height_array = band_dom_data[i:i + 768, j:j + 768]
                cv2.imwrite(
                    os.path.join(out_dir, os.path.basename(dom_path).split('.')[0] + str(i) + '_' + str(j) + '.png'),
                    dsm_array)
                np.save(
                    os.path.join(out_dir, os.path.basename(dom_path).split('.')[0] + str(i) + '_' + str(j) + '.npy'),
                    height_array)
        # writeTiff(band_dom_data, band_dom.XSize, band_dom.YSize, 1, domgeoTransform, im_proj, 'demo.tiff')
        # data.append(band_data)
        # print("band " + (i + 1).__str__() + " read success.")
        # return data


if __name__ == '__main__':
    dom_data = 'D:\workspace\dom'
    dsm_data = 'D:\workspace\dsm'
    out_path = 'D:\workspace\ddata'
    for dom_file in os.listdir(dom_data):
        if 'tif' not in dom_file:
            continue
        for dsm_file in os.listdir(dsm_data):
            if 'tif' not in dsm_file:
                continue
            readImage(os.path.join(dom_data, dom_file),
                      os.path.join(dsm_data, dsm_file),
                      out_path)
