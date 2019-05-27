#!/usr/bin/env python3

import gi

import imutils

import time
import numpy as np
import cv2
import IPython
import io

from edgetpu.detection.engine import DetectionEngine
from PIL import Image
from PIL import ImageDraw

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject


# Function to read labels from text files.
def ReadLabelFile(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    ret = {}
    for line in lines:
        pair = line.strip().split(maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret

class SensorFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, **properties):
        super(SensorFactory, self).__init__(**properties)
        self.cap = cv2.VideoCapture('rkisp device=/dev/video1 io-mode=4 ! video/x-raw,format=NV12,width=640,height=480,framerate=30/1 ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
        self.number_frames = 0
        self.fps = 30
        self.duration = 1 / self.fps * Gst.SECOND  # duration of a frame in nanoseconds
        self.launch_string = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                             'caps=video/x-raw,format=BGR,width=640,height=480,framerate={}/1 ' \
                             '! videoconvert ! video/x-raw,format=I420 ' \
                             '! mpph264enc ' \
                             '! rtph264pay config-interval=1 name=pay0 pt=96'.format(self.fps)
        self.engine = DetectionEngine('/home/pi/TPU-MobilenetSSD/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite')
        self.labels = ReadLabelFile('/home/pi/TPU-MobilenetSSD/coco_labels.txt') 
        


    def on_need_data(self, src, lenght):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                prepimg = frame[:, :, ::-1].copy()
                prepimg = Image.fromarray(prepimg)
                draw = ImageDraw.Draw(prepimg)
                tinf = time.perf_counter()
                #print("Hello detect !!!!")
                t1 = time.time()
                out = self.engine.DetectWithImage(prepimg, threshold=0.5, keep_aspect_ratio=True, relative_coord=False, top_k=10)
                if out:
                    for obj in out:
                        #print ('-----------------------------------------')
                        #if labels:
                        #    print(labels[obj.label_id])
                        #print ('score = ', obj.score)
                        box = obj.bounding_box.flatten().tolist()
                        #print ('box = ', box)
                        # Draw a rectangle.
                        draw.rectangle(box, outline='red')
                        if self.labels:
                            draw.text((box[0] + (box[2]-box[0]), box[1]), self.labels[obj.label_id] , fill='green')

                t2 = time.time()
                fps = 1/(t2-t1)
                fps_str = 'FPS = %.2f' % fps
                draw.text((10,220), fps_str , fill='green')
                
                imcv = np.asarray(prepimg)[:,:,::-1].copy()
                
                data = imcv.tostring()
                buf = Gst.Buffer.new_allocate(None, len(data), None)
                buf.fill(0, data)
                buf.duration = self.duration
                timestamp = self.number_frames * self.duration
                buf.pts = buf.dts = int(timestamp)
                buf.offset = timestamp
                self.number_frames += 1
                retval = src.emit('push-buffer', buf)
                print('pushed buffer, frame {}, duration {} ns, durations {} s'.format(self.number_frames,
                                                                                       self.duration,
                                                                                       self.duration / Gst.SECOND))
                if retval != Gst.FlowReturn.OK:
                    print(retval)

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        self.number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)


class GstServer(GstRtspServer.RTSPServer):
    def __init__(self, **properties):
        super(GstServer, self).__init__(**properties)
        self.factory = SensorFactory()
        self.factory.set_shared(True)
        self.get_mount_points().add_factory("/test", self.factory)
        self.attach(None)


GObject.threads_init()
Gst.init(None)

server = GstServer()

loop = GObject.MainLoop()
loop.run()