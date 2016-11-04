"""Utils for writing html debug files.

Mostly those methods used by convnet02.py
"""

import base64
import io
import os

import numpy as np
from PIL import Image
import tensorflow as tf

import convnetshared1 as convshared


def argmax(l):
    max = -1000000000.0
    index = 0
    for i in l:
        if i > max:
            max = index
        index = index + 1
    return max


def write_html_image_tensor_gray(outfile, tensor, rgb, scale=1):
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
    outfile.write("<div style='border:2px;border-style:solid;border-color:#66a;margin-top:2px;padding:2px'>")
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
    outfile.write('<img style ="image-rendering:pixelated" width="' + str(w*scale) + '" src="data:image/png;base64,')
    # outfile.write('<img style ="image-rendering:pixelated" src="data:image/png;base64,')
    outfile.write(b64)
    outfile.write('" alt="testImage.png"><br/>')
    outfile.write('min: ' + str(min) + '<br/>')
    outfile.write('max: ' + str(max) + '<br/>')
    outfile.write('</div>')


def write_vertical_meter(outfile, x, total, col = 'rgb(255, 255, 0)'):
    outfile.write('<svg width = "8" height = "' + str(total*8) + '" style="background:#606060"><rect width = "7" height = "' + str(x*8)  + '" y = "' + str((total-x) * 8) + '" style = "fill:' + col + ';" /></svg>')


def write_html_image(outfile, result, result_throttle, images, answers, answers_throttle, w, h, message):
    color = "style='background:#9f1010'"
    if argmax(answers) == result:
        color = "style='background:#a0ffa0'"
    if (argmax(answers) == result - 1) or (argmax(answers) == result + 1):
        color = "style='background:#dfaf10'"
    if (argmax(answers) == result - 2) or (argmax(answers) == result + 2):
        color = "style='background:#bf5f10'"
    outfile.write('<td ' + color + '>')
    # tempImg = np.add(images, 0.5)
    # tempImg = np.multiply(tempImg, 255.0)
    tempImg = np.copy(images)
    b64 = Image.frombuffer('RGB', (w, h), tempImg.astype(np.int8), 'raw', 'RGB', 0, 1)
    # b64.save("testtest.png")
    b = io.BytesIO()
    b64.save(b, 'PNG')
    b64 = base64.b64encode(b.getvalue())
    outfile.write('<img src="data:image/png;base64,')
    outfile.write(b64)
    outfile.write('" alt="testImage.png">')
    total = convshared.max_log_outs
    throttle_net = result_throttle
    throttle_gt = argmax(answers_throttle)
    # outfile.write('<svg width = "10" height = "' + str(total*8) + '" style="background:#606060"><rect width = "10" height = "' + str(throttle_gt*8)  + '" y = "' + str((total-throttle_gt) * 8) + '" style = "fill:rgb(255,255,0);" /></svg>' + str(throttle_gt))
    # write_vertical_meter(outfile, throttle_gt, total, 'rgb(40, 255, 40)')
    # write_vertical_meter(outfile, throttle_net, total)
    for i in range(argmax(answers_throttle)):
        outfile.write('T')
    outfile.write('&nbsp;&nbsp;' + str(argmax(answers_throttle)) + '<br/>')
    for i in range(result_throttle):
        outfile.write('N')
    outfile.write('<br/>')

    for i in range(argmax(answers)):
        outfile.write('*')
    outfile.write('&nbsp;&nbsp;' + str(argmax(answers)) + '<br/>')
    for i in range(result):
        outfile.write('N')
    outfile.write('&nbsp;&nbsp;' + str(result))
    outfile.write('<br/>')
    outfile.write(message)
    outfile.write('</td>')


def write_html(output_path, results, results_throttle, images, answers, answers_throttles, odos, w, h, graph, testImages, sess, test_feed_dict):
    # images = [x for (y,x) in sorted(zip(results,images), key=lambda pair: pair[0])]
    outfile = open(os.path.join(output_path, "debug.html"), "w")
    outfile.write("""
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <title>title</title>
    <!--    <link rel="stylesheet" href="style.css">
        <script src="script.js"></script>-->
      </head>
      <body>
        <!-- page content -->
        <table style="background: #333;font-family: monospace;">
                  """)
    outfile.write('<tr>')

    # variables = graph.get_collection('htmlize')
    variables = graph.get_collection(tf.GraphKeys.VARIABLES)

    name_to_var = {}
    for var in variables:
        if var.name:
            name_to_var[var.name] = var

    for i in xrange(len(results)):
        if (i % 16) == 0:
            outfile.write('</tr>')
            outfile.write('<tr>')
        write_html_image(outfile, results[i], results_throttle[i], images[i], answers[i], answers_throttles[i], w, h, str(odos[i][0]*1000.0))
    outfile.write('</tr>')
    outfile.write('</table>')
    write_html_image(
        outfile, results[0], results_throttle[0], testImages[0], answers[0],
        answers_throttles[0], w, h, str(odos[0][0]*1000.0))
    # write_html_image_RGB(outfile, all_xs[0], width, height)
    # viz = sess.run(W)
    # write_html_image_RGB(outfile, viz, width, height)

    results = sess.run(name_to_var['W_conv1:0'])
    results = results.transpose(3,0,1,2) # different because RGB
    write_html_image_tensor_gray(outfile, results, True, 4)
    results = sess.run(name_to_var['W_conv2:0'])
    results = results.transpose(2,3,0,1)
    write_html_image_tensor_gray(outfile, results, False, 4)
    # results = sess.run(name_to_var['W_fc1:0'])
    # results = results.transpose(1,0)
    # results = results.reshape((convshared.fc1_num_outs, convshared.l5_num_convs, convshared.heightD32, convshared.widthD32))
    # results = results.transpose(0, 2, 1, 3)
    # results = results.reshape((convshared.fc1_num_outs, convshared.heightD32, convshared.l5_num_convs * convshared.widthD32))
    # write_html_image_tensor_gray(outfile, results, False)
    # # write_html_histogram(outfile, results)
    # results = sess.run(name_to_var['W_fc2:0'])
    # write_html_image_tensor_gray(outfile, results, False)
    # results = sess.run(name_to_var['W_fc3:0'])
    # write_html_image_tensor_gray(outfile, results, False)
    results = sess.run(convshared.h_pool5_odo_concat, feed_dict=test_feed_dict)
    write_html_image_tensor_gray(outfile, results, False)

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
    outfile.write("""
        WRITTEN!!!!
      </body>
    </html>
                  """)
    outfile.close()
