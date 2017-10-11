"""Utils for writing html debug files.

Mostly those methods used by convnet02.py
"""

import base64
import io
from shutil import copyfile
import os

import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

from cStringIO import StringIO
import PIL

# http://stackoverflow.com/questions/4931376/generating-matplotlib-graphs-without-a-running-x-server
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import config

num_conv_debugs = 4

def argmax(l):
    max = -1000000000.0
    index = 0
    for i in l:
        if i > max:
            max = index
        index = index + 1
    return max

class HtmlDebug:
    def __init__(self):
        self.buffer = []
        self.write_html_header()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def write_file(self, output_path):
        self.write_html_end()
        outfile = open(os.path.join(output_path, "debug.html"), "w")
        outfile.write(''.join(self.buffer))
        outfile.close()

    def write_html_header(self):
        self.buffer.append("""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>title</title>
    <style type="text/css">
      body{
        font-family:monospace;
      }
      ul.listbutton li span{
        color: #475dba;
        float: left;
        display:inline-block;
        background-color:#ebebeb;
        border:1px solid #b3cbef;
        margin-right: 4px;
        padding:4px;
        text-align: center;
        line-height: 24px;
        text-decoration: none;
        cursor:pointer;
        user-select: none;
      }
      ul.listbutton li span:hover {
        text-decoration: none;
        color: #000000;
        background-color: #33B5E5;
      }
      ul.listbutton li span:active {
        text-decoration: none;
        color: #000000;
        background-color: #f3B5E5;
      }
    </style>
  </head>
  <body onload="myMove()">""")

    def write_html_end(self):
        self.buffer.append("""
        </body>
    </html>""")

    def write_line(self, s):
        self.buffer.append(str(s) + '<br/>')

    def write_html_image_tensor_gray(self, tensor, rgb, scale=1, label=None):
        dims = len(tensor.shape)
        h = tensor.shape[dims - 2]
        w = tensor.shape[dims - 1]
        d = 1
        if dims >= 3:
            d = tensor.shape[dims - 3]
        b = 1
        if dims >= 4:
            b = tensor.shape[dims - 4]
        if rgb == True:
            d = 1
            b = 1
            w = tensor.shape[dims - 3]
            h = tensor.shape[dims - 2]
            if dims >= 4:
                d = tensor.shape[dims - 4]
        self.buffer.append("<div style='border:2px;border-style:solid;border-color:#66a;margin-top:2px;padding:2px'>")
        if label: self.write_line(label)
        tempImg = tensor.flatten()
        # range expand image to [0..255]
        min = np.amin(tempImg)
        max = np.amax(tempImg)
        tempImg = np.add(tempImg, -min)
        tempImg = np.multiply(tempImg, 255.0)
        tempImg = np.divide(tempImg, max - min)

        if rgb == True:
            b64 = Image.frombuffer('RGB', (w, h*d), tempImg.astype(np.int8), 'raw', 'RGB', 0, 1)
        else:
            b64 = Image.frombuffer('L', (w, h*b*d), tempImg.astype(np.int8), 'raw', 'L', 0, 1)
        # b64.save("testtest.png")
        b = io.BytesIO()
        b64.save(b, 'PNG')
        b64 = base64.b64encode(b.getvalue())
        self.buffer.append('<img style ="image-rendering:-moz-crisp-edges;image-rendering:pixelated" width="' + str(w*scale) + '" src="data:image/png;base64,')
        # self.buffer.append('<img style ="image-rendering:pixelated" src="data:image/png;base64,')
        self.buffer.append(b64)
        self.buffer.append('" alt="testImage.png"><br/>\n')
        self.buffer.append('min: ' + str(min) + '<br/>')
        self.buffer.append('max: ' + str(max) + '<br/>')
        self.buffer.append('</div>')


    def write_html_image_tensor_gray_overlay(self, tensor, scale, layer_id, im_id, conv_max):
        dims = len(tensor.shape)
        h = tensor.shape[dims - 2]
        w = tensor.shape[dims - 1]
        d = 1
        if dims >= 3:
            d = tensor.shape[dims - 3]
        b = 1
        if dims >= 4:
            b = tensor.shape[dims - 4]
        # self.buffer.append('<div id="pixels' + str(layer_id) + '_' + str(im_id) + '" style="display:none;position:absolute;top:2px;left:2px;">')
        self.buffer.append('<div id="pixels' + str(layer_id) + '_' + str(im_id) + '" style="position:absolute;top:2px;left:132px;">')
        tempImg = tensor#.flatten()
        # range expand image to [0..255]
        min = 0.0#np.amin(tempImg)
        max = conv_max# np.amax(tempImg)
        tempImg = np.add(tempImg, -min)
        tempImg = np.multiply(tempImg, 16.0)
        tempImg = np.clip(tempImg, 0.0, 255.0)
        # tempImg = np.divide(tempImg, max - min)
        tempImg = tempImg.astype(np.int8)
        newImg = np.zeros((tensor.shape[0], tensor.shape[1], 4), dtype=np.int8)
        newImg[:,:,0] = 255
        newImg[:,:,1] = 0
        newImg[:,:,2] = 64
        newImg[:,:,3] = tempImg

        b64 = Image.frombuffer('RGBA', (w, h), newImg, 'raw', 'RGBA', 0, 1)
        # b64 = Image.frombuffer('RGBA', (w, h*b*d), tempImg, 'raw', 'L', 0, 1)
        # b64.save("testtest.png")
        b = io.BytesIO()
        b64.save(b, 'PNG')
        b64 = base64.b64encode(b.getvalue())
        self.buffer.append('<img style ="image-rendering:-moz-crisp-edges;image-rendering:pixelated" width="' + str(w*scale) + '" src="data:image/png;base64,')
        # self.buffer.append('<img style ="image-rendering:pixelated" src="data:image/png;base64,')
        self.buffer.append(b64)
        self.buffer.append('">\n')
        self.buffer.append("<div style='position:absolute;top:0px;left:0px;color:#8f8'>" + str(max) + "</div>")

        self.buffer.append('</div>')


    def write_vertical_meter(self, x, total, col = 'rgb(255, 255, 0)'):
        self.buffer.append('<svg width = "8" height = "' + str(total*8) + '" style="background:#606060"><rect width = "7" height = "' + str(x*8)  + '" y = "' + str((total-x) * 8) + '" style = "fill:' + col + ';" /></svg>')

    def write_steering_line(self, x, col = 'rgb(255, 255, 0)', line_width = 3):
        s = '<svg width="190" height="160" stroke-width="%s" style="position:absolute;top:0px;left:0px"><path d="M64 128 Q 64 84 %s 66" stroke="%s" fill="transparent"/></svg>' % (line_width, str(x + 64), col)
        self.buffer.append(s)

    def encode_image_as_html(self, img, filetype='JPEG', attrib =''):
        # img.save("testtest.png")
        b = io.BytesIO()
        img.save(b, filetype, quality=50)
        img = base64.b64encode(b.getvalue())
        self.buffer.append('<img ' + attrib + ' src="data:image/png;base64,')
        self.buffer.append(img)
        self.buffer.append('">\n')

    def draw_softmax_distribution(self, label, softmax, gt, draw_zero_point=False):
        soft_img = Image.new('RGBA', (128, 32), (0, 0, 0, 64))
        draw = ImageDraw.Draw(soft_img)
        soft_size = softmax.shape[0]
        scale = (soft_img.width / soft_size)
        # draw rectangle to mark 0-speed mapping
        if draw_zero_point:
            draw.rectangle([5 * scale, soft_img.height / 2, 5 * scale + scale - 2, soft_img.height], (16, 16, 32, 56))
        for i in range(soft_size):
            fill_color = (255, 255, 255, 255)
            if i == gt: fill_color = (0, 255, 0, 255)
            prob = softmax[i]
            draw.rectangle([i * scale, soft_img.height - round(prob * soft_img.height), i * scale + scale - 2, soft_img.height], fill_color)
        self.buffer.append('<div style="position:relative;padding-bottom:4px">')
        self.encode_image_as_html(soft_img, 'PNG', 'style="position:absolute"')
        self.buffer.append(label + '</div><br/>')

    def write_html_image(self, test_data, w, h, message, im_id, im_id_adjust, steering_regress, throttle_regress):
        images = test_data.pic_array[im_id_adjust]
        steer_gt = test_data.steer_array[im_id_adjust]
        throttle_gt = test_data.throttle_array[im_id_adjust]

        # pulse_entropy = np.sum( np.multiply(pulse_softmax, np.log(np.reciprocal(pulse_softmax))) )

        # Fade color from green to yellow to red and make a table cell with that background color.
        # (40, 88, 136, 184, 232, 280)
        # (400,352,304, 256, 208, 160, 112, 64, 16)
        # delta = abs(steering_gt - result_steering)
        # shade = 'rgb(%s, %s, %s)' % (min(232, delta * 52 + 20), max(0, min(255-32, 448 - delta * 48)), min(80, delta * 80))
        #remap entropy to around [0..1] range.
        # ent01 = min(1.0, max(0.0, pulse_entropy * 0.5 - 0.1))
        # shade = 'rgb(%s, %s, %s)' % (min(255,int(ent01 * 255*2)), max(0,int((2.0-ent01*2.0) * 255)), 64)
        # in_set = im_id in max_sets[0]
        # if in_set:
        shade = 'rgb(255, 255, 255)'
        color = "style='background:%s;position:relative;padding:2px;border:2px solid black;white-space:nowrap;'" % (shade)
        padded_id = str(im_id).zfill(5)
        self.buffer.append('<td id="td' + padded_id + '" ' + color + '><span>')

        # Save out camera image as embedded .png and draw steering direction curves on the image.
        tempImg = np.copy(images)
        b64 = Image.frombuffer('RGB', (w, h), tempImg.astype(np.int8), 'raw', 'RGB', 0, 1)
        self.encode_image_as_html(b64)
        #self.buffer.append('<img src="track_extents_white.png"><svg height="128" width="128" style="position:absolute;top:2px;left:138px"><circle cx="' + str(results_lon * 128 / 15 + 7) + '" cy="' + str(results_lat * 128 / 15 + 7) + '" r="4" stroke="black" stroke-width="1" fill="red" /></svg>')
        #write_html_image_tensor_gray_overlay(outfile, latlon_softmax.reshape((15, 15))*255, 9, "", "", 1.0)
        #self.buffer.append('<svg height="128" width="128" style="position:absolute;top:2px;left:138px"><circle cx="' + str(gt_lon * 128 / 15 + 7) + '" cy="' + str(gt_lat * 128 / 15 + 7) + '" r="4" stroke="black" stroke-width="1" fill="yellow" /></svg>')
        self.write_steering_line(-steering_regress * 1, 'rgb(240, 55, 40)', 7)
        self.write_vertical_meter(throttle_gt, 16, 'rgb(40, 255, 40)')
        self.write_vertical_meter(throttle_regress, 16)
        self.buffer.append('</span>')

        # Print out throttle, steering, and odometer values.
        self.buffer.append('<br/>regress: ' + str(int(steering_regress)) + ' ' + str(int(throttle_regress)) + '<br/>')
        self.buffer.append('gt: ' + str(int(steer_gt)) + ' ' + str(int(throttle_gt)) + '<br/>')
        # self.buffer.append('ent: ' + str(pulse_entropy))
        self.buffer.append('</td>')


    def draw_graph(self, arrs, title=None):
        plt.figure(figsize=(30, 6), dpi=100)
        assert len(arrs) <= 3  # only 3 colors defined
        line_styles = ['r-', 'g-', 'b-']
        html_styles = ['red', 'green','blue']
        for counter, arr in enumerate(arrs):
            std = np.std(arr)
            mean = np.mean(arr)
            self.buffer.append('<span style="color:' + html_styles[counter] + '">mean: ' + str(mean) + '&nbsp;&nbsp;&nbsp;std: ' + str(std) + '</span><br/>')
            plt.plot(range(len(arr)), arr, line_styles[counter])
        # plt.plot(range(numTest - net_model.n_steps)[50:plt_len], results_throttle_regress[50:plt_len], 'r-')
        # plt.plot(range(numTest - net_model.n_steps)[50:plt_len], test_data.throttle_array[50:plt_len], 'g-')
        # axes = plt.gca()
        # axes.set_ylim([0, 1000.05])
        if title: plt.title(title)
        # plt.xlabel('iteration')
        # plt.ylabel('diff squared')
        # plt.savefig(os.path.join(output_path, "progress.png"))
        # plt.imshow(np.random.random((20, 20)))
        buffer_ = StringIO()
        plt.savefig(buffer_, format="png", bbox_inches='tight')
        buffer_.seek(0)
        image = PIL.Image.open(buffer_)
        # ar = np.asarray(image)
        self.encode_image_as_html(image, filetype='PNG', attrib='')
        buffer_.close()
        plt.close('all')
        self.buffer.append('<br/>')


    def write_html(self, test_data, graph, sess, results_steering_regress, results_throttle_regress, net_model, feed_dict):
        # copyfile("track_extents_white.png", os.path.join(output_path, "track_extents_white.png"))
        image_count = len(results_steering_regress)
        image_count = min(1000, image_count)

        self.buffer.append("""
        <div id="topdiv">
          <br/>
          <ul id="buttons" class="listbutton" style="list-style-type:none;padding:0;margin:0;">
            <li><span id="listbutton0" onclick="listButtonClick(-1)">None</span></li>
            """)

        for i in xrange(num_conv_debugs):
            self.buffer.append('<li><span id="listbutton' + str(i + 1) + '" onclick="listButtonClick(' + str(i) + ')">' + str(i) + '</span></li>')

        self.buffer.append("""
          </ul>
          <br style="clear:both" />
        </div>
        <br/>
        <table id="mainTable" onclick="tableClick()" style="border-collapse:collapse;background: #333;font-family: monospace;">
                      """)
        self.buffer.append('<tr>')

        # variables = graph.get_collection('htmlize')
        variables = graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        name_to_var = {}
        for var in variables:
            if var.name:
                name_to_var[var.name] = var

        # Make a giant table of all images and info from the neural net.
        for i in xrange(image_count):
            if (i % 16) == 0:
                self.buffer.append('</tr>')
                self.buffer.append('<tr>')
            self.write_html_image(test_data, config.width, config.height, "blank", i, i + net_model.n_steps - 1,
                             results_steering_regress[i], results_throttle_regress[i])
        self.buffer.append('</tr>')
        self.buffer.append('</table><br/><br/>')

        self.buffer.append("<div style='position:relative'>")
        self.write_html_image(
            test_data, config.width, config.height, "blank2", 0, net_model.n_steps - 1,
            results_steering_regress[0], results_throttle_regress[0])
        self.buffer.append('</div>')
        # write_html_image_RGB(outfile, all_xs[0], width, height)
        # viz = sess.run(W)
        # write_html_image_RGB(outfile, viz, width, height)

        for key, value in net_model.visualizations.iteritems():
            results = sess.run(value[1], feed_dict=feed_dict)
            if value[0] == 'gray_batch_steps':
                results = results[0]#.reshape((net_model.n_steps, config.width, config.height, config.img_channels))
                # results = results.transpose(3,0,1,2) # different because RGB
                self.write_html_image_tensor_gray(results, False, 4, label=str(key))
            if value[0] == 'rgb_batch_steps':
                results = results[0]#.reshape((net_model.n_steps, config.width, config.height, config.img_channels))
                # results = results.transpose(3,0,1,2) # different because RGB
                self.write_html_image_tensor_gray(results, True, 1, label=str(key))
            if value[0] == 'rgb_batch':
                # results = results[0].reshape((config.width, config.height, config.img_channels))
                # results = results.transpose(3,0,1,2) # different because RGB
                self.write_html_image_tensor_gray(results[0], True, 1, label=str(key))

        # results = sess.run(name_to_var['shared_conv/W_conv1:0'])
        # results = results.transpose(3,0,1,2) # different because RGB
        # write_html_image_tensor_gray(outfile, results, True, 4)
        # results = sess.run(name_to_var['shared_conv/W_conv2:0'])
        # results = results.transpose(2,3,0,1)
        # write_html_image_tensor_gray(outfile, results, False, 4)
        # results = sess.run(name_to_var['shared_cnn/W_fc1:0'])
        # results = results.transpose(1,0)
        # results = results.reshape((NNModel.fc1_num_outs, NNModel.l4_num_convs, NNModel.heightD32, NNModel.widthD32))
        # results = results.transpose(0, 2, 1, 3)
        # results = results.reshape((NNModel.fc1_num_outs, NNModel.heightD32, NNModel.l5_num_convs * NNModel.widthD32))
        # write_html_image_tensor_gray(outfile, results, False)
        # # write_html_histogram(outfile, results)
        # results = sess.run(name_to_var['W_fc2:0'])
        # write_html_image_tensor_gray(outfile, results, False)
        # results = sess.run(name_to_var['W_fc3:0'])
        # write_html_image_tensor_gray(outfile, results, False)

        # results = sess.run(net_model.h_pool5_flat, feed_dict=test_data.FeedDict(net_model))
        # write_html_image_tensor_gray(outfile, results, False)

        # results = sess.run(h_pool1, feed_dict={x: all_xs[0:1], y_: all_ys[0:1], keep_prob: 1.0})
        # # results = results.transpose(0,3,1,2)
        # results = results.transpose(3,0,1,2)
        # write_html_image_tensor_gray(outfile, results, False, 2)
        # results = sess.run(h_pool2, feed_dict={x: all_xs[0:1], y_: all_ys[0:1], keep_prob: 1.0})
        # # results = results.transpose(0,3,1,2)
        # results = results.transpose(3,0,1,2)
        # write_html_image_tensor_gray(outfile, results, False, 2)
        # results = sess.run(h_pool3, feed_dict={x: all_xs[0:1], y_: all_ys[0:1], keep_prob: 1.0})
        # results = results.transpose(3,0,1,2)
        # write_html_image_tensor_gray(outfile, results, False, 2)
        # results = sess.run(h_pool4, feed_dict={x: all_xs[0:1], y_: all_ys[0:1], keep_prob: 1.0})
        # results = results.transpose(3,0,1,2)
        # write_html_image_tensor_gray(outfile, results, False, 2)
        # results = sess.run(h_pool5, feed_dict={x: all_xs[0:1], y_: all_ys[0:1], keep_prob: 1.0})
        # results = results.transpose(3,0,1,2)
        # write_html_image_tensor_gray(outfile, results, False, 2)
        # results = sess.run(y_conv, feed_dict={x: all_xs[0:1], y_: all_ys[0:1], keep_prob: 1.0})
        # write_html_image_tensor_gray(outfile, results, False, 8)
        self.buffer.append("""
        WRITTEN!!!!
    
      <script>
        var button = document.createElement("input");
        button.type = "button";
        button.value = "Animate";
        button.addEventListener ("click", tableClick);
        document.body.insertBefore(button, document.getElementById("mainTable"));
    
        var animating = false;
        var pos = 0;
        var elemCount = NUM_IMAGES;
        var timerID = 0;
        function pad(num, size) {
          var s = "000000000" + num;
          return s.substr(s.length-size);
        }
        function myMove() {
          if (timerID == 0) {
            timerID = setInterval(frame, 133);
          }
          function frame() {
            if (!animating) {
              return;
            }
            if (pos == elemCount * 4) {
              clearInterval(timerID);
              timerID = 0;
            } else {
              pos++;
              var modA = pos % elemCount
              var modB = (pos + 1) % elemCount
              //console.log(" " + pos + "   td" + pad(modA, 5) + "    " + "td" + pad(modB, 5));
              var elem = document.getElementById("td" + pad(modA, 5));
              elem.style.display = "none";
              elem = document.getElementById("td" + pad(modB, 5));
              elem.style.display = "table-cell";
            }
          }
        }
        function tableClick() {
          if (animating) {
            animating = !animating;
            clearInterval(timerID);
            timerID = 0;
            pos = 0;
            for (i = 0; i < elemCount; i++) {
              elem = document.getElementById("td" + pad(i, 5));
              elem.style.display = "table-cell";
            }
          } else {
            pos = 0;
            for (i = 0; i < elemCount; i++) {
              elem = document.getElementById("td" + pad(i, 5));
              elem.style.display = "none";
            }
            animating = !animating;
            myMove();
          }
        }
        function listButtonClick(index) {
          var ei;
          for (ei = 0; ei < elemCount; ei++) {
            var i;
            for (i = 0; i < num_conv_debugs; i++) {
              var elems = document.getElementById("pixels" + i + "_" + ei);
              elems.style.display = "none";
              if (i == index) elems.style.display = "block";
            }
          }
          for (i = -1; i < num_conv_debugs; i++) {
            var button = document.getElementById("listbutton" + (i + 1));
            if (i == index) {
              button.style.backgroundColor = "#2385b5";
            } else {
              button.style.backgroundColor = "";
            }
          }
        }
      </script>
    
                      """.replace("NUM_IMAGES", str(image_count)).replace("num_conv_debugs", str(num_conv_debugs)))
