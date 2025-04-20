"""
将osm文件转化为csv文件
Wed Dec  1 2021
Copyright (c) 2021 Yuzhen FENG
"""
import osm2gmns as og

net = og.getNetFromFile("./primitive/data.osm")
og.consolidateComplexIntersections(net)
og.outputNetToCSV(net, output_folder='primitive')
